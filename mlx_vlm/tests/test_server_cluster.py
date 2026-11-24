# Copyright © 2026 Neywa Labs Inc.

import asyncio
import json
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)

import mlx_vlm.server as server
from mlx_vlm.server import cluster as management
from mlx_vlm.server.cluster import ClusterManager
from mlx_vlm.server.cluster_types import (
    Members,
    Memory,
    NodeIdentity,
    PrepareModel,
    RegisterNode,
    SSHTransport,
    StartGroup,
    StopGroup,
    UpdateNode,
)
from mlx_vlm.server.distributed import Worker
from mlx_vlm.server.distributed_launch import WorkerLauncher, managed_workers
from mlx_vlm.server.runtime import runtime

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def info(node_id, rank=0, **updates):
    value = NodeIdentity(
        id=node_id,
        name="Same display name",
        physical_id="physical-" + node_id,
        os="test",
        mlx_version="test",
        mlx_vlm_version="test",
        memory=Memory(total_bytes=1024, used_bytes=512, mlx_active_bytes=123),
        measured_at=time.time(),
        backends=["jaccl", "ring"],
        worker={
            "rank": rank,
            "world_size": 2,
            "protocol_version": 2,
            "backend": "jaccl",
            "group_id": "g1",
            "instance_id": node_id + "-process",
        },
        model_dir="/different/" + node_id + "/models",
    )
    return value.model_copy(update=updates)


@pytest.fixture
def control(tmp_path, monkeypatch):
    value = ClusterManager(tmp_path)
    monkeypatch.setattr(runtime, "cluster_manager", value)
    monkeypatch.setattr(runtime, "distributed_coordinator", None)
    monkeypatch.setattr(runtime, "distributed_worker", None)
    monkeypatch.setattr(management, "identity", lambda *_: info("coordinator"))
    monkeypatch.setenv("MLX_VLM_SERVER_API_KEY", "public-secret")

    async def probe(endpoint, credential):
        assert credential == "worker-secret"
        node_id = endpoint.rsplit("/", 1)[-1].split("-")[0]
        return info(node_id, 0 if node_id == "a" else 1)

    value.probe = probe
    yield value
    if value.transfers is not None:
        value.transfers.close()


@pytest.fixture
def own_launcher(monkeypatch):
    def create(coordinator):
        coordinator.launcher = WorkerLauncher(coordinator)
        monkeypatch.setattr(coordinator.launcher, "_stop", Mock())
        return coordinator.launcher

    return create


async def register_pair(control):
    for node_id in ("a", "b"):
        await control.register(
            RegisterNode(
                id=node_id,
                endpoint="http://" + node_id,
                api_key="worker-secret",
                ssh=SSHTransport(
                    destination="worker-" + node_id,
                    python="/different/" + node_id + "/venv/bin/python",
                ),
            )
        )
    await control.select(Members(node_ids=["a", "b"]))


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "launch", "fast_sync": 1},
        {"mode": "launch", "fast_sync": 0},
        {"mode": "launch", "fast_sync": "true"},
        {"mode": "launch", "fast_sync": "false"},
        {"mode": "attach", "fast_sync": True},
        {"mode": "attach", "fast_sync": False},
    ],
)
def test_start_api_rejects_invalid_or_external_fast_sync(control, body):
    client = TestClient(server.app)
    response = client.post(
        "/v1/cluster/start",
        headers={"Authorization": "Bearer public-secret"},
        json=body,
    )
    assert response.status_code == 422
    assert not control.operations and not control.tasks


@pytest.mark.parametrize("configured", [False, True])
async def test_start_api_toggles_fast_sync_without_changing_default(
    control, monkeypatch, configured
):
    from dataclasses import replace

    from mlx_vlm.server.distributed_launch import LaunchConfig

    await register_pair(control)
    initial = management.Coordinator(
        replace(
            control.config(),
            launch=LaunchConfig(
                control.root / "hosts.json", str(control.root), fast_sync=configured
            ),
        )
    )
    monkeypatch.setattr(runtime, "distributed_coordinator", initial)
    control.bootstrapped = True
    start = AsyncMock()
    monkeypatch.setattr(WorkerLauncher, "start", start)
    monkeypatch.setattr(WorkerLauncher, "_stop", Mock())
    monkeypatch.setattr(management.Coordinator, "_health", AsyncMock())
    for override in (True, False, None):
        body = StartGroup(mode="launch", fast_sync=override)
        operation = await control.begin("start", body, f"start-{override}")
        await asyncio.gather(*control.tasks)
        assert operation.status == "succeeded", operation.error
        assert start.call_args.kwargs["fast_sync"] is override
        assert runtime.distributed_coordinator.config.launch.fast_sync is configured
        active = runtime.distributed_coordinator
        calls = start.await_count
        repeated = await control.begin("start", body, f"start-{override}")
        assert repeated.id == operation.id and start.await_count == calls
        rejected = await control.begin(
            "start", StartGroup(mode="launch", fast_sync=not override), None
        )
        await asyncio.gather(*control.tasks)
        assert rejected.error.code == "already_running"
        assert runtime.distributed_coordinator is active
        assert start.await_count == calls
        stopped = await control.begin("stop", StopGroup(), None)
        await asyncio.gather(*control.tasks)
        assert stopped.status == "succeeded", stopped.error
        assert control.snapshot().fast_sync is None
    assert initial.config.launch.fast_sync is configured


@pytest.mark.parametrize("configured", [False, True])
def test_cluster_state_reports_effective_fast_sync(control, monkeypatch, configured):
    from dataclasses import replace

    from mlx_vlm.server.distributed_launch import LaunchConfig

    coordinator = management.Coordinator(
        replace(
            control.config(),
            launch=LaunchConfig(
                control.root / "hosts.json", str(control.root), fast_sync=configured
            ),
        )
    )
    monkeypatch.setattr(runtime, "distributed_coordinator", coordinator)
    control.nodes["a"] = NS(identity=info("a"))
    control.running = ["a"]
    assert control.snapshot().fast_sync is None  # Externally managed group.
    coordinator.launcher = WorkerLauncher(coordinator)
    coordinator.launcher.launch = replace(
        coordinator.config.launch, fast_sync=not configured
    )
    assert control.snapshot().fast_sync is not configured
    control.running = []
    assert control.snapshot().fast_sync is None


async def test_authenticated_identity_preserving_updates_and_persistence(control):
    await register_pair(control)
    node = await control.update(
        "a", UpdateNode(endpoint="http://a-new", name="renamed")
    )
    assert node.id == "a" and node.identity.physical_id == "physical-a"
    assert len(control.nodes) == 2 and control.selected == ["a", "b"]
    with pytest.raises(HTTPException) as error:
        await control.update("a", UpdateNode(endpoint="http://b"))
    assert error.value.detail["code"] == "identity_mismatch"
    restored = ClusterManager(control.root)
    assert restored.nodes["a"].endpoint == "http://a-new"
    assert restored.nodes["a"].stale
    assert restored.running == []
    assert (control.root / "registry.json").stat().st_mode & 0o777 == 0o600
    assert "worker-secret" not in node.model_dump_json()


async def test_registration_verifies_approved_id_and_disallows_duplicates(control):
    with pytest.raises(HTTPException) as error:
        await control.register(
            RegisterNode(
                id="approved", endpoint="http://other", api_key="worker-secret"
            )
        )
    assert error.value.detail["code"] == "identity_mismatch"
    await register_pair(control)
    control.probe = AsyncMock(return_value=info("new", physical_id="physical-a"))
    with pytest.raises(HTTPException) as error:
        await control.register(
            RegisterNode(id="new", endpoint="http://new", api_key="worker-secret")
        )
    assert error.value.detail["code"] == "duplicate_device"


def test_api_auth_openapi_and_idempotent_delete(control, monkeypatch):
    client = TestClient(server.app)
    assert client.get("/v1/cluster/nodes").status_code == 401
    headers = {"Authorization": "Bearer public-secret"}
    result = client.post(
        "/v1/cluster/nodes",
        headers=headers,
        json={"id": "a", "endpoint": "http://a", "api_key": "worker-secret"},
    )
    assert result.status_code == 200, result.text
    assert "worker-secret" not in result.text
    assert client.delete("/v1/cluster/nodes/a", headers=headers).status_code == 204
    assert client.delete("/v1/cluster/nodes/a", headers=headers).status_code == 204
    schema = client.get("/openapi.json").json()
    assert "RegisterNode" in schema["components"]["schemas"]
    assert "Operation" in schema["components"]["schemas"]
    assert "/v1/cluster/events" in schema["paths"]
    monkeypatch.delenv("MLX_VLM_SERVER_API_KEY")
    assert client.get("/v1/cluster").status_code == 503


async def test_membership_immutable_while_running_and_backend_checks(control):
    await register_pair(control)
    with pytest.raises(HTTPException) as error:
        await control.select(Members(node_ids=["a", "missing"]))
    assert error.value.detail["code"] == "unknown_members"
    with pytest.raises(HTTPException) as error:
        await control.select(Members(node_ids=["a", "b"], backend="nccl"))
    assert error.value.detail["code"] == "backend_unavailable"
    control.running = ["a", "b"]
    with pytest.raises(HTTPException) as error:
        await control.update("a", UpdateNode(endpoint="http://a-new"))
    assert error.value.detail["code"] == "group_busy"
    with pytest.raises(HTTPException):
        await control.select(Members(node_ids=["b", "a"]))


async def test_memory_only_selected_fresh_nodes_and_no_coordinator_double_count(
    control,
):
    await register_pair(control)
    aggregate = control.snapshot().memory
    assert aggregate.total_bytes == 2048
    assert aggregate.used_bytes == 1024
    assert aggregate.physical_devices == 2
    control.nodes["b"].identity.measured_at -= 30
    assert control.snapshot().memory.stale_node_ids == ["b"]
    assert not control.snapshot().memory.complete
    control.nodes["b"].identity.memory.used_bytes = None
    assert control.snapshot().memory.missing_node_ids == ["b"]


async def test_operations_serialized_idempotent_and_external_stop_does_not_kill(
    control, monkeypatch
):
    await register_pair(control)
    entered, release = asyncio.Event(), asyncio.Event()
    original = control.stop

    async def delayed(operation, body):
        entered.set()
        await release.wait()
        await original(operation, body)

    monkeypatch.setattr(control, "stop", delayed)
    first = await control.begin("stop", StopGroup(), "retry-key")
    await entered.wait()
    repeated = await control.begin("stop", StopGroup(), "retry-key")
    assert repeated.id == first.id
    with pytest.raises(HTTPException) as error:
        await control.begin("start", StartGroup(), "other")
    assert error.value.detail["code"] == "management_busy"
    with pytest.raises(HTTPException) as error:
        await control.begin("start", StartGroup(), "retry-key")
    assert error.value.detail["code"] == "idempotency_conflict"
    release.set()
    await asyncio.gather(*control.tasks)
    assert first.status == "succeeded" and control.state == "stopped"
    assert control.events[-2].data["external_processes_stopped"] is False
    restored = ClusterManager(control.root)
    assert restored.operations[first.id].status == "succeeded"


@pytest.mark.parametrize("imported", [False, True])
async def test_rejected_start_preserves_owned_group(
    control, monkeypatch, imported, own_launcher
):
    await register_pair(control)
    coordinator = management.Coordinator(control.config())
    launcher = own_launcher(coordinator)
    monkeypatch.setattr(runtime, "distributed_coordinator", coordinator)
    control.bootstrapped = not imported
    if not imported:
        control.state, control.running = "ready", ["a", "b"]
    monitor = asyncio.create_task(asyncio.Event().wait())
    launcher.monitor_task = monitor
    try:
        for key in ("start-again", "start-again", "new-start"):
            operation = await control.begin("start", StartGroup(mode="launch"), key)
            await asyncio.gather(*control.tasks)
            assert operation.status == "failed"
            assert operation.error.code == "already_running"
            assert control.snapshot().state == "ready"
            assert control.running == ["a", "b"]
            assert runtime.distributed_coordinator is coordinator
            assert not coordinator.draining and not coordinator.failure.exists()
            assert launcher.monitor_task is monitor and not monitor.done()
            launcher._stop.assert_not_called()
        stopped = await control.begin("stop", StopGroup(), "stop")
        await asyncio.gather(*control.tasks)
        assert stopped.status == "succeeded"
        launcher._stop.assert_called_once()
        assert monitor.done() and coordinator.launcher is None
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)


@pytest.mark.parametrize("interruption", ["retained", "timeout", "cancel"])
async def test_rejected_stop_restores_healthy_group(
    control, monkeypatch, interruption, own_launcher
):
    await register_pair(control)
    coordinator = management.Coordinator(control.config())
    monkeypatch.setattr(runtime, "distributed_coordinator", coordinator)
    launcher = own_launcher(coordinator)
    control.state, control.running = "ready", ["a", "b"]
    coordinator.continuations["paused"] = {"session_id": "retained-session"}
    if interruption != "retained":
        coordinator.active = 1
    operation = await control.begin("stop", StopGroup(drain_timeout_seconds=0.01), None)
    tasks = list(control.tasks)
    if interruption == "cancel":
        while not coordinator.draining:
            await asyncio.sleep(0)
        tasks[0].cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert operation.status == "failed"
    assert (
        operation.error.code
        == {
            "retained": "retained_continuations",
            "timeout": "drain_timeout",
            "cancel": "transition_failed",
        }[interruption]
    )
    assert control.snapshot().state == "ready"
    assert control.running == ["a", "b"] and coordinator.launcher is launcher
    assert not coordinator.draining and not coordinator.failure.exists()
    assert coordinator.continuations == {"paused": {"session_id": "retained-session"}}
    launcher._stop.assert_not_called()
    # The rejected stop leaves sessions usable and a later explicit discard valid.
    coordinator.active = 0
    coordinator.expire_continuations = AsyncMock()
    retry = await control.begin("stop", StopGroup(discard_continuations=True), None)
    await asyncio.gather(*control.tasks)
    assert retry.status == "succeeded", retry.error
    coordinator.expire_continuations.assert_awaited_once()
    launcher._stop.assert_called_once()


async def test_failed_checkpoint_disposal_keeps_admissions_blocked(
    control, monkeypatch, own_launcher
):
    await register_pair(control)
    coordinator = management.Coordinator(control.config())
    monkeypatch.setattr(runtime, "distributed_coordinator", coordinator)
    control.state, control.running = "ready", ["a", "b"]
    launcher = own_launcher(coordinator)
    coordinator.continuations["paused"] = {}
    coordinator.expire_continuations = AsyncMock(side_effect=RuntimeError("rank lost"))
    operation = await control.begin("stop", StopGroup(discard_continuations=True), None)
    await asyncio.gather(*control.tasks)
    assert operation.status == "failed"
    assert control.state == "failed" and coordinator.draining
    assert control.running == ["a", "b"]
    launcher._stop.assert_not_called()


async def test_node_failure_during_drain_is_not_rolled_back(control, monkeypatch):
    await register_pair(control)
    coordinator = management.Coordinator(control.config())
    monkeypatch.setattr(runtime, "distributed_coordinator", coordinator)
    control.state, control.running = "ready", ["a", "b"]
    coordinator.active = 1
    operation = await control.begin("stop", StopGroup(drain_timeout_seconds=0.01), None)
    while not coordinator.draining:
        await asyncio.sleep(0)
    coordinator.failed(RuntimeError("worker lost while draining"))
    await asyncio.gather(*control.tasks)
    assert operation.status == "failed"
    assert operation.error.code == "drain_timeout"
    assert control.state == "failed" and coordinator.draining
    assert coordinator.failure.exists()


@pytest.mark.parametrize("failure_phase", ["launch", "health"])
async def test_failed_start_cleans_up_only_its_new_launcher(
    control, monkeypatch, failure_phase, own_launcher
):
    from dataclasses import replace

    from mlx_vlm.server import distributed_launch

    await register_pair(control)
    prior = management.Coordinator(
        replace(control.config(), launch=NS(backend="jaccl"))
    )
    monkeypatch.setattr(runtime, "distributed_coordinator", prior)
    control.bootstrapped = True
    previous_launcher = own_launcher(prior)
    launched = []

    def create(coordinator):
        launcher = own_launcher(coordinator)
        launcher.start = AsyncMock(
            side_effect=(
                RuntimeError("launch failed") if failure_phase == "launch" else None
            )
        )
        launched.append(launcher)
        return launcher

    monkeypatch.setattr(distributed_launch, "WorkerLauncher", create)
    monkeypatch.setattr(
        management.Coordinator,
        "_health",
        AsyncMock(side_effect=RuntimeError("health failed")),
    )
    operation = await control.begin("start", StartGroup(mode="launch"), None)
    await asyncio.gather(*control.tasks)
    assert operation.status == "failed" and control.state == "failed"
    assert runtime.distributed_coordinator is prior and not control.running
    assert prior.launcher is previous_launcher
    previous_launcher._stop.assert_not_called()
    launched[0]._stop.assert_called_once()


async def test_api_restarts_cli_group_and_shutdown_closes_only_owned_workers(
    control, monkeypatch
):
    from dataclasses import replace

    await register_pair(control)
    rdma = {
        "local_address": "192.0.2.1",
        "local_device": "rdma_en4",
        "remote_device": "rdma_en5",
    }
    config = control.config()
    initial = management.Coordinator(
        replace(
            config,
            workers=(config.workers[0], replace(config.workers[1], rdma=rdma)),
            launch=NS(backend="jaccl"),
        )
    )
    monkeypatch.setattr(runtime, "distributed_coordinator", initial)
    monkeypatch.setattr(management.Coordinator, "_health", AsyncMock())
    stop = Mock()
    monkeypatch.setattr(WorkerLauncher, "_stop", stop)

    async def start(launcher, **kwargs):
        launcher.monitor_task = asyncio.create_task(asyncio.Event().wait())

    monkeypatch.setattr(WorkerLauncher, "start", start)
    async with managed_workers(initial):
        await control.refresh(force=True)
        assert control.nodes["a"].ssh.rdma is None
        assert control.nodes["b"].ssh.rdma.model_dump() == rdma
        restored = ClusterManager(control.root)
        assert [worker.rdma for worker in restored.config().workers] == [None, rdma]
        first = initial.launcher
        stopped = await control.begin("stop", StopGroup(), None)
        await asyncio.gather(*control.tasks)
        assert stopped.status == "succeeded", stopped.error
        assert first.closed and first.monitor_task.done()
        assert initial.launcher is None
        stop.assert_called_once()
        restarted = await control.begin("start", StartGroup(mode="launch"), None)
        await asyncio.gather(*control.tasks)
        assert restarted.status == "succeeded", restarted.error
        assert runtime.distributed_coordinator.config.workers[1].rdma == rdma
        second = runtime.distributed_coordinator.launcher
        assert second is not first and not second.monitor_task.done()
        assert control.snapshot().ownership == "server"
        await control.close()
        assert second.closed and second.monitor_task.done()
    # The CLI lifespan still holds its original launcher, already stopped by API.
    assert stop.call_count == 2


async def test_external_attach_validates_group_then_exposes_rank_zero(
    control, monkeypatch
):
    async def health(coordinator, client):
        coordinator.group_instances = ["a-process", "b-process"]

    monkeypatch.setattr(management.Coordinator, "_health", health)

    await register_pair(control)
    operation = await control.begin("start", StartGroup(), "start")
    await asyncio.gather(*control.tasks)
    assert operation.status == "succeeded", operation.error
    state = control.snapshot()
    assert state.inference_coordinator_id == "a"
    assert state.running_members == ["a", "b"]
    assert state.ownership == "external"
    assert (
        runtime.distributed_coordinator.config.workers[1].model_dir
        == "/different/b/models"
    )
    operation = await control.begin("stop", StopGroup(), "stop")
    await asyncio.gather(*control.tasks)
    assert operation.status == "succeeded"
    assert runtime.distributed_coordinator.draining


@pytest.mark.parametrize("status", [200, 503])
async def test_loading_progress_is_visible_before_relay_finishes(
    control, monkeypatch, status
):
    await register_pair(control)
    monkeypatch.setattr(control, "check_model", AsyncMock())
    monkeypatch.setattr(management.Coordinator, "_health", AsyncMock())
    monkeypatch.setattr(
        control, "stage_model", AsyncMock(return_value="/models/digest")
    )
    release = asyncio.Event()

    async def relay(*args, **kwargs):
        await release.wait()
        return {}, status

    def health(request):
        return httpx.Response(
            200,
            json={
                "distributed_worker": {
                    "load_progress": {
                        "model": "/models/digest",
                        "stage": "evaluating_layers",
                        "completed_layers": 3,
                        "total_layers": 47,
                        "elapsed_seconds": 1.2,
                    }
                }
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(management.Coordinator, "relay", relay)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(**kw, transport=httpx.MockTransport(health)),
    )
    operation = await control.begin("start", StartGroup(model="org/model"), None)

    async def wait_for_loading():
        while operation.phase != "loading_model":
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(wait_for_loading(), timeout=3)
        assert operation.status == "running"
        progress = operation.result["progress"]
        assert progress["workers_ready"] == 0 and progress["total_workers"] == 2
        assert [n["node_id"] for n in progress["nodes"]] == ["a", "b"]
        assert all(n["completed_layers"] == 3 for n in progress["nodes"])
        assert control.events[-1].data["result"]["progress"] == progress
    finally:
        release.set()
        await asyncio.gather(*control.tasks)
    assert operation.status == ("succeeded" if status == 200 else "failed")
    assert "progress" in operation.result  # Keep measurements in the terminal result.


async def test_loading_warmup_does_not_block_worker_health(control, monkeypatch):
    from mlx_vlm.server import openai

    entered, release = threading.Event(), threading.Event()

    def loading(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        raise HTTPException(400, "test checkpoint unavailable")

    monkeypatch.setattr(runtime, "distributed_worker", NS(info=lambda: {}))
    monkeypatch.setattr(openai, "get_cached_model", loading)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app),
        base_url="http://server",
        headers={"Authorization": "Bearer public-secret"},
    ) as client:
        task = asyncio.create_task(
            client.post(
                "/v1/responses/input_tokens",
                json={"model": "test", "input": "hello", "distributed": True},
                headers={"X-MLX-Distributed-Request": "a" * 32},
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert not task.done()
            assert (await client.get("/health")).status_code == 200
            assert not task.done()
        finally:
            release.set()
            response = await task
        assert response.status_code == 400


async def test_external_reconfiguration_is_observable_without_launching(control):
    await register_pair(control)
    control.probe = AsyncMock(return_value=info("a", worker=None))
    operation = await control.begin("start", StartGroup(), None)
    await asyncio.gather(*control.tasks)
    assert operation.status == "failed"
    assert operation.error.code == "external_reconfiguration_required"
    assert operation.result["members"][0]["rank"] == 0
    assert runtime.distributed_coordinator is None and control.state == "stopped"


@pytest.mark.parametrize(
    "result, failed",
    [
        (OSError("offline"), True),
        (httpx.ReadTimeout("busy loading"), False),
        (info("changed-identity"), True),
    ],
)
async def test_telemetry_failure_preserves_local_loader_and_distinguishes_timeouts(
    control, monkeypatch, result, failed
):
    await register_pair(control)
    coordinator = management.Coordinator(control.config())
    monkeypatch.setattr(runtime, "distributed_coordinator", coordinator)
    local = runtime.response_generator
    control.running = ["a", "b"]
    control.state = "ready"
    healthy_probe = control.probe
    control.probe = (
        AsyncMock(side_effect=result)
        if isinstance(result, Exception)
        else AsyncMock(return_value=result)
    )
    await control.refresh(force=True)
    expected_state = "failed" if failed else "ready"
    assert control.state == expected_state
    assert coordinator.failure.exists() == failed
    assert runtime.response_generator is local
    assert all(not node.reachable and node.stale for node in control.nodes.values())
    if not failed:
        assert control.nodes["a"].error.code == "telemetry_timeout"
    control.probe = healthy_probe
    await control.refresh(force=True)
    assert all(node.reachable and not node.stale for node in control.nodes.values())
    assert control.state == expected_state
    assert coordinator.failure.exists() == failed


async def test_events_replay_and_reset_after_retention_gap(control):
    api = management.router(lambda: None)
    endpoint = next(
        route.endpoint for route in api.routes if route.path == "/v1/cluster/events"
    )
    await control.emit("node.registered", {"id": "a"})
    checkpoint = control.event_id
    await control.emit("node.updated", {"id": "a"})
    response = await endpoint(last_event_id=checkpoint, interval_seconds=1)
    first = await anext(response.body_iterator)
    assert "node.updated" in first and "node.registered" not in first
    await response.body_iterator.aclose()
    response = await endpoint(last_event_id=0, interval_seconds=1)
    assert "cluster.snapshot" in await anext(response.body_iterator)
    await response.body_iterator.aclose()


@pytest.mark.parametrize("config_file", ["config.json", "params.json"])
async def test_hub_model_preflight_accepts_both_config_names(
    control, monkeypatch, tmp_path, config_file
):
    config = {"model_type": "mistral_large4"}
    path = tmp_path / config_file
    path.write_text(json.dumps(config))
    download = Mock(
        side_effect=(
            [str(path)]
            if config_file == "config.json"
            else [
                RemoteEntryNotFoundError(
                    "missing config.json", response=NS(headers={}, request=None)
                ),
                str(path),
            ]
        )
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    validate = Mock()
    monkeypatch.setattr(management, "validate_model", validate)
    control.selected = ["a", "b"]
    await control.check_model("org/model", "revision-a")
    assert [call.args[1] for call in download.call_args_list] == (
        ["config.json"]
        if config_file == "config.json"
        else ["config.json", "params.json"]
    )
    assert all(
        call.kwargs == {"revision": "revision-a"} for call in download.call_args_list
    )
    validate.assert_called_once_with(config, 2)


@pytest.mark.parametrize(
    "error",
    [
        RepositoryNotFoundError("unauthorized", response=NS(headers={}, request=None)),
        LocalEntryNotFoundError("offline"),
        HfHubHTTPError("server error", response=NS(headers={}, request=None)),
    ],
)
async def test_hub_model_preflight_does_not_hide_download_errors(
    control, monkeypatch, error
):
    download = Mock(side_effect=error)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(type(error)):
        await control.check_model("org/model", None)
    download.assert_called_once_with("org/model", "config.json", revision=None)


def test_model_member_count_errors_are_actionable(monkeypatch):
    import mlx_vlm.utils as utils

    monkeypatch.setattr(
        utils, "get_model_and_args", lambda *_: (NS(Model=NS(shard=lambda: None)), None)
    )
    config = {
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 96,
    }
    management.validate_model(config, 2)
    with pytest.raises(HTTPException) as error:
        management.validate_model(config, 3)
    assert error.value.detail["code"] == "unsupported_member_count"
    assert (
        "16" in error.value.detail["message"]
        and "3 workers" in error.value.detail["message"]
    )
    assert error.value.detail["action"]


@pytest.mark.parametrize("world_size", [2, 3, 5, 32])
def test_inkling2_member_count_uses_expert_ownership(world_size):
    config = {
        "model_type": "inkling2",
        "text_config": {
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "n_routed_experts": 32,
        },
    }
    management.validate_model(config, world_size)
    with pytest.raises(HTTPException) as error:
        management.validate_model(config, 33)
    assert error.value.detail["code"] == "unsupported_member_count"


async def test_weight_delivery_operation_is_independent_of_membership(
    control, monkeypatch
):
    def prepare(
        stager, model, revision, *, validate_sharding, progress, transport, stop
    ):
        assert not validate_sharding
        assert transport == "ssh_rsync" and not stop.is_set()
        assert len(stager.config.workers) == 1
        worker = stager.config.workers[0]
        assert worker.ssh == "worker-b"
        assert worker.python == "/different/b/venv/bin/python"
        assert worker.api_key == "worker-secret"
        progress("transferring", node_id=worker.node_id, total_bytes=123)
        path = str(stager.config.model_dir / "digest")
        stager.worker_paths[(path, worker.url)] = "/different/b/models/digest"
        return path

    monkeypatch.setattr(management.ModelStager, "prepare", prepare)

    await register_pair(control)
    control.running = ["a", "b"]
    control.state = "ready"
    op = await control.begin(
        "prepare_model", PrepareModel(model="org/model", node_ids=["b"]), "delivery"
    )
    await asyncio.gather(*control.tasks)
    assert op.status == "succeeded", op.error
    assert op.result["nodes"] == [
        {
            "id": "b",
            "path": "/different/b/models/digest",
            "transport": "ssh_rsync",
            "verified": True,
        }
    ]
    assert control.running == ["a", "b"] and control.state == "ready"
    assert any(e.data.get("phase") == "transferring" for e in control.events)
    retry = await control.begin(
        "prepare_model", PrepareModel(model="org/model", node_ids=["b"]), "delivery"
    )
    assert retry.id == op.id


@pytest.mark.parametrize("kind", ["prepare_model", "start"])
async def test_transfer_bytes_are_observable_while_copy_is_still_running(
    control, monkeypatch, kind
):
    entered, release = threading.Event(), threading.Event()

    def prepare(stager, model, revision, *, progress, **options):
        for count in (25, 50):
            progress(
                "transferring",
                node_id="b",
                snapshot="digest",
                bytes_transferred=count,
                bytes_reused=0,
                total_bytes=100,
            )
        entered.set()
        assert release.wait(5)
        path = str(stager.config.model_dir / "digest")
        for worker in stager.config.workers:
            stager.worker_paths[(path, worker.url)] = (
                "/different/" + worker.node_id + "/models/digest"
            )
        return path

    monkeypatch.setattr(management.ModelStager, "prepare", prepare)
    monkeypatch.setattr(control, "check_model", AsyncMock())
    monkeypatch.setattr(management.Coordinator, "_health", AsyncMock())
    monkeypatch.setattr(
        management.Coordinator, "relay", AsyncMock(return_value=({}, 200))
    )

    await register_pair(control)
    api = management.router(lambda: None)
    endpoints = {route.path: route.endpoint for route in api.routes}
    cursor = control.event_id
    body = (
        PrepareModel(model="org/model", node_ids=["b"])
        if kind == "prepare_model"
        else StartGroup(model="org/model")
    )
    operation = await control.begin(kind, body, None)
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        status = await endpoints["/v1/cluster/operations/{operation_id}"](operation.id)
        assert status.status == "running" and status.phase == "transferring"
        assert status.result["progress"] == dict(
            node_id="b",
            snapshot="digest",
            bytes_transferred=50,
            bytes_reused=0,
            total_bytes=100,
        )
        response = await endpoints["/v1/cluster/events"](
            last_event_id=cursor, interval_seconds=1
        )
        observed = []
        try:
            while len(observed) < 2:
                wire = await asyncio.wait_for(anext(response.body_iterator), 2)
                event = json.loads(
                    next(
                        line[6:]
                        for line in wire.splitlines()
                        if line.startswith("data: ")
                    )
                )
                if (
                    event["type"] == "operation.updated"
                    and event["data"]["phase"] == "transferring"
                ):
                    observed.append(event["data"]["result"]["progress"])
        finally:
            await response.body_iterator.aclose()
        assert [data["bytes_transferred"] for data in observed] == [25, 50]
        assert all(
            data["node_id"] == "b" and data["total_bytes"] == 100 for data in observed
        )
        assert operation.status == "running"
    finally:
        release.set()
        await asyncio.gather(*control.tasks)
    assert operation.status == "succeeded", operation.error
    if kind == "start":
        assert operation.result["state"] == "ready"
        assert operation.result["members"] == ["a", "b"]
        assert operation.result["progress"]["total_workers"] == 2


async def test_failed_delivery_does_not_fail_running_inference_group(
    control, monkeypatch
):
    def broken(*args, **kwargs):
        raise OSError("rsync interrupted")

    monkeypatch.setattr(management.ModelStager, "prepare", broken)

    await register_pair(control)
    control.state = "ready"
    control.running = ["a", "b"]
    op = await control.begin("prepare_model", PrepareModel(model="org/model"), None)
    await asyncio.gather(*control.tasks)
    assert op.status == "failed" and "rsync interrupted" in op.error.message
    assert control.state == "ready" and control.running == ["a", "b"]


async def test_delivery_requires_explicit_existing_ssh_access_for_remote_devices(
    control,
):
    await register_pair(control)
    control.nodes["b"].ssh = None
    with pytest.raises(HTTPException) as error:
        control.config()
    assert error.value.detail["code"] == "ssh_transport_required"
    assert "outside MLX-VLM" in error.value.detail["action"]


def test_no_http_weight_upload_api_and_delivery_is_authenticated(control):
    client = TestClient(server.app)
    assert (
        client.post(
            "/v1/cluster/models/prepare", json={"model": "org/model"}
        ).status_code
        == 401
    )
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/cluster/models/prepare" in paths
    assert not any("/staging/" in path for path in paths)
    assert control.snapshot().capabilities.checkpoint_transfer == [
        "ssh_rsync",
        "jaccl_rdma",
    ]


@pytest.mark.parametrize("selected_transport", ["auto", "ssh_rsync", "jaccl_rdma"])
async def test_cancel_preparation_is_authenticated_idempotent_and_keeps_inference(
    control, monkeypatch, selected_transport
):
    entered = threading.Event()

    def prepare(stager, model, revision, *, stop, transport, **options):
        assert transport == selected_transport
        entered.set()
        assert stop.wait(5)
        raise InterruptedError("Checkpoint preparation cancelled")

    monkeypatch.setattr(management.ModelStager, "prepare", prepare)

    await register_pair(control)
    control.state, control.running = "ready", ["a", "b"]
    op = await control.begin(
        "prepare_model",
        PrepareModel(model="org/model", transport=selected_transport),
        "cancel-test",
    )
    assert await asyncio.to_thread(entered.wait, 3)
    routes = {r.path: r.endpoint for r in management.router(lambda: None).routes}
    cancel = routes["/v1/cluster/operations/{operation_id}/cancel"]
    assert (await cancel(op.id)).id == op.id
    await asyncio.gather(*control.tasks)
    assert op.status == "failed" and op.error.code == "preparation_cancelled"
    assert (await cancel(op.id)).id == op.id
    assert control.state == "ready" and control.running == ["a", "b"]
    await control.close()

    client = TestClient(server.app)
    assert client.post("/v1/cluster/operations/any/cancel").status_code == 401


@pytest.mark.parametrize(
    "endpoint",
    [
        "ftp://worker",
        "http:///worker",
        "http://user:secret@worker",
        "http://worker/path",
        "http://worker?query",
        "http://worker#fragment",
    ],
)
def test_worker_endpoints_have_same_validation_in_config_and_api(endpoint):
    for create in (
        lambda: Worker(endpoint),
        lambda: RegisterNode(id="worker", endpoint=endpoint, api_key="key"),
        lambda: UpdateNode(endpoint=endpoint),
    ):
        with pytest.raises(ValueError, match="origin"):
            create()


@pytest.mark.parametrize("endpoint", ["http://worker/", "https://[::1]:8283/"])
def test_worker_endpoints_normalize_identically_in_config_and_api(endpoint):
    worker = Worker(endpoint)
    registration = RegisterNode(id="worker", endpoint=endpoint, api_key="key")
    update = UpdateNode(endpoint=endpoint)
    assert worker.url == registration.endpoint == update.endpoint == endpoint[:-1]


@pytest.mark.parametrize("destination", ["-oProxyCommand=bad", "host;cmd", "host name"])
def test_ssh_destinations_have_same_validation_in_config_and_api(destination):
    for create in (
        lambda: Worker("http://worker", ssh=destination),
        lambda: SSHTransport(destination=destination),
    ):
        with pytest.raises(ValueError):
            create()


@pytest.mark.parametrize("path", ["relative", "/tmp/space here", "/tmp/x;touch bad"])
def test_ssh_worker_staging_paths_reject_shell_metacharacters(path):
    with pytest.raises(ValueError, match="model_dir"):
        Worker("http://worker", ssh="worker", model_dir=path)


async def test_external_attach_rejects_empty_group_identity(control):
    await register_pair(control)

    async def probe(endpoint, credential):
        node_id = endpoint.rsplit("/", 1)[-1]
        value = info(node_id, 0 if node_id == "a" else 1)
        value.worker["group_id"] = ""
        return value

    control.probe = probe
    operation = await control.begin("start", StartGroup(), None)
    await asyncio.gather(*control.tasks)
    assert operation.status == "failed"
    assert operation.error.code == "group_identity_mismatch"
    assert not control.running


async def test_owned_launch_does_not_silently_ignore_selected_backend(
    control, monkeypatch
):
    from dataclasses import replace

    await register_pair(control)
    config = replace(control.config(), launch=NS(backend="ring"))
    monkeypatch.setattr(
        runtime, "distributed_coordinator", management.Coordinator(config)
    )
    operation = await control.begin("start", StartGroup(mode="launch"), None)
    await asyncio.gather(*control.tasks)
    assert operation.status == "failed"
    assert operation.error.code == "launch_reconfiguration_required"
    assert runtime.distributed_coordinator.launcher is None


async def test_failed_group_can_stop_without_contacting_dead_session_endpoints(
    control, monkeypatch, own_launcher
):
    await register_pair(control)
    coordinator = management.Coordinator(control.config())
    coordinator.group_instances = ["old-a", "old-b"]
    coordinator.continuations["dead-session"] = {}
    coordinator.expire_continuations = AsyncMock(
        side_effect=AssertionError("dead ranks cannot acknowledge disposal")
    )
    coordinator.failed(RuntimeError("rank lost"))
    monkeypatch.setattr(runtime, "distributed_coordinator", coordinator)
    control.running = ["a", "b"]
    launcher = own_launcher(coordinator)
    operation = await control.begin("stop", StopGroup(discard_continuations=True), None)
    await asyncio.gather(*control.tasks)
    assert operation.status == "succeeded", operation.error
    launcher._stop.assert_called_once()
    assert coordinator.failure.exists() and not coordinator.continuations
    assert not control.running and control.state == "stopped"
    assert operation.result["collective_disposal_acknowledged"] is False
    assert operation.result["external_restart_required"] is False


@pytest.mark.parametrize("bundle", [False, True])
async def test_start_stages_and_reports_speculative_draft(
    control, monkeypatch, tmp_path, bundle
):
    from mlx_vlm.server.cluster_types import SpeculativeDraft

    await register_pair(control)
    monkeypatch.setattr(control, "check_model", AsyncMock())
    monkeypatch.setattr(management.Coordinator, "_health", AsyncMock())
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model_type": "inkling2_dspark"}')
    stage = AsyncMock(return_value=str(snapshot))
    monkeypatch.setattr(control, "stage_model", stage)
    monkeypatch.setattr(control, "load_model", AsyncMock(return_value=({}, 200)))
    draft = SpeculativeDraft(
        model="org/target" if bundle else "org/draft", block_size=3
    )
    operation = await control.begin(
        "start", StartGroup(model="org/target", draft=draft), None
    )
    await asyncio.gather(*control.tasks)
    assert operation.status == "succeeded", operation.error
    assert stage.await_count == (1 if bundle else 2)
    if not bundle:
        assert stage.call_args.args[2:4] == ("org/draft", None)
        assert stage.call_args.kwargs == {"validate_sharding": False}
    coordinator = runtime.distributed_coordinator
    assert coordinator.draft_target == ("org/target", None)
    assert coordinator.draft_path == str(snapshot)
    assert control.snapshot().draft == draft.model_copy(update={"kind": "dflash"})


@pytest.mark.parametrize(
    "body",
    [
        {"draft": {"model": "org/draft"}},
        {"model": "org/target", "draft": {"model": "org/draft", "block_size": 1}},
        {"model": "org/target", "draft": {"model": "org/draft", "kind": "eagle3"}},
    ],
)
def test_start_rejects_invalid_speculative_configuration(body):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        StartGroup.model_validate(body)


def test_bundled_draft_inherits_pinned_target_revision():
    body = StartGroup.model_validate(
        {"model": "org/target", "revision": "pinned", "draft": {"model": "org/target"}}
    )
    assert body.draft.revision == "pinned"
    explicit = StartGroup.model_validate(
        {
            "model": "org/target",
            "revision": "pinned",
            "draft": {"model": "org/target", "revision": "different"},
        }
    )
    assert explicit.draft.revision == "different"


async def test_auto_detected_unsupported_draft_never_loads_workers(
    control, monkeypatch, tmp_path
):
    await register_pair(control)
    monkeypatch.setattr(control, "check_model", AsyncMock())
    monkeypatch.setattr(management.Coordinator, "_health", AsyncMock())
    (tmp_path / "config.json").write_text('{"model_type": "eagle3"}')
    monkeypatch.setattr(control, "stage_model", AsyncMock(return_value=str(tmp_path)))
    load = AsyncMock()
    monkeypatch.setattr(control, "load_model", load)
    operation = await control.begin(
        "start",
        StartGroup.model_validate(
            {"model": "org/target", "draft": {"model": "org/draft"}}
        ),
        None,
    )
    await asyncio.gather(*control.tasks)
    assert operation.status == "failed" and operation.error.code == "unsupported_draft"
    load.assert_not_called()
    assert runtime.distributed_coordinator is None
