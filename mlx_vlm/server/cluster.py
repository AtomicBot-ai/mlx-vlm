# Copyright © 2026 Neywa Labs Inc.
"""Authenticated registry and lifecycle controller for the inference coordinator.

Discovery, consent, and external process launch are intentionally outside this
module. An endpoint is trusted only after its authenticated identity is checked.
"""

import asyncio
import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import replace
from pathlib import Path

import httpx
from fastapi import APIRouter, Header, HTTPException, Query
from starlette.responses import StreamingResponse

from .cluster_identity import identity, state_directory
from .cluster_types import (
    AggregateMemory,
    Capabilities,
    ClusterError,
    ClusterErrorResponse,
    ClusterEvent,
    ClusterState,
    Members,
    Node,
    NodeIdentity,
    Operation,
    PrepareModel,
    RegisterNode,
    SSHTransport,
    StartGroup,
    StopGroup,
    UpdateNode,
)
from .distributed import ClusterConfig, Coordinator, ModelStager, Worker
from .runtime import runtime


def fail(code, message, status=409, *, action=None, retryable=False):
    raise HTTPException(
        status,
        ClusterError(
            code=code, message=message, action=action, retryable=retryable
        ).model_dump(),
    )


def validate_model(config, count):
    """Conservative preflight; architecture-specific validators can refine it."""
    from ..utils import get_model_and_args

    architecture, _ = get_model_and_args(config)
    if not callable(getattr(architecture.Model, "shard", None)):
        fail(
            "model_not_shardable",
            "This model has no distributed shard implementation.",
            422,
        )
    validator = getattr(architecture, "validate_sharding", None)
    if validator is not None:
        try:
            validator(config, count)
        except ValueError as error:
            fail(
                "unsupported_member_count",
                str(error),
                422,
                action="Select a supported worker count for this model.",
            )
        return
    text = config.get("text_config") or config
    for field in (
        "num_attention_heads",
        "num_key_value_heads",
        "num_attention_groups",
        "intermediate_size",
        "moe_intermediate_size",
    ):
        dimension = text.get(field)
        if isinstance(dimension, int) and dimension > 0 and dimension % count:
            fail(
                "unsupported_member_count",
                f"{field}={dimension} cannot be evenly sharded across {count} workers.",
                422,
                action="Select a divisor of the model dimensions, or use an architecture with an explicit validate_sharding implementation.",
            )


class ClusterManager:
    def __init__(self, root=None):
        self.root = Path(root) if root else state_directory()
        self.nodes = {}
        self.credentials = {}
        self.selected = []
        self.running = []
        self.backend = None
        self.state = "stopped"
        self.lock = asyncio.Lock()
        self.operations = {}
        self.idempotency = {}
        self.current = None
        self.tasks = set()
        self.events = deque(maxlen=1024)
        self.event_id = int(time.time() * 1000000)
        self.changed = asyncio.Condition()
        self.last_refresh = 0
        self.bootstrapped = False
        self.transfers = None
        self.transfer_stops = {}
        path = self.root / "registry.json"
        if path.exists():
            data = json.loads(path.read_text())
            self.nodes = {
                v["id"]: Node.model_validate(v) for v in data.get("nodes", [])
            }
            self.credentials = data.get("credentials", {})
            self.selected, self.backend = data.get("selected", []), data.get("backend")
            # Never assume a group survives a coordinator restart. Reattach explicitly.
            for node in self.nodes.values():
                node.stale = True
            for value in data.get("operations", []):
                operation = Operation.model_validate(value)
                if operation.status == "running":
                    operation.status = "failed"
                    operation.phase = "interrupted"
                    operation.error = ClusterError(
                        code="coordinator_restarted",
                        message="Operation interrupted; inspect workers and retry.",
                        retryable=True,
                    )
                self.operations[operation.id] = operation
            self.idempotency = data.get("idempotency", {})

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.root / (".registry-" + uuid.uuid4().hex)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            json.dump(
                {
                    "nodes": [v.model_dump() for v in self.nodes.values()],
                    "credentials": self.credentials,
                    "selected": self.selected,
                    "backend": self.backend,
                    "operations": [v.model_dump() for v in self.operations.values()],
                    "idempotency": self.idempotency,
                },
                output,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(path, self.root / "registry.json")

    async def emit(self, kind, data):
        async with self.changed:
            self.event_id += 1
            self.events.append(
                ClusterEvent(
                    id=self.event_id, type=kind, timestamp=time.time(), data=data
                )
            )
            self.changed.notify_all()

    def idle(self):
        if self.current and self.operations[self.current].status == "running":
            fail(
                "management_busy",
                "Another management operation is running.",
                action=f"Observe /v1/cluster/operations/{self.current}",
                retryable=True,
            )

    def mutable(self, node_id=None):
        self.idle()
        coordinator = runtime.distributed_coordinator
        if (self.running and (node_id is None or node_id in self.running)) or (
            coordinator and (coordinator.active or coordinator.waiters)
        ):
            fail(
                "group_busy",
                "Drain and stop the group before changing its membership or endpoints.",
                action="POST /v1/cluster/stop",
                retryable=True,
            )

    async def probe(self, endpoint, credential):
        async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
            response = await client.get(
                endpoint + "/internal/distributed/identity",
                headers={"Authorization": "Bearer " + credential},
            )
            response.raise_for_status()
            return NodeIdentity.model_validate(response.json())

    async def register(self, request):
        async with self.lock:
            self.idle()
            try:
                info = await self.probe(
                    request.endpoint, request.api_key.get_secret_value()
                )
            except Exception:
                fail(
                    "identity_unreachable",
                    "Authenticated worker identity verification failed.",
                    502,
                    retryable=True,
                )
            if info.id != request.id:
                fail(
                    "identity_mismatch",
                    "Endpoint did not present the approved node ID.",
                    422,
                )
            old = self.nodes.get(request.id)
            if old and old.identity.physical_id != info.physical_id:
                fail(
                    "identity_mismatch",
                    "A registered ID cannot be reassigned to another device.",
                    422,
                )
            if old and old.endpoint != request.endpoint:
                fail(
                    "already_registered",
                    "Use PATCH to change a registered node's endpoint.",
                )
            if any(
                n.id != request.id
                and (
                    n.endpoint == request.endpoint
                    or n.identity.physical_id == info.physical_id
                )
                for n in self.nodes.values()
            ):
                fail(
                    "duplicate_device",
                    "This endpoint or physical device is already registered.",
                )
            if (
                old
                and old.ssh == request.ssh
                and (not request.name or old.name == request.name)
                and self.credentials[request.id] == request.api_key.get_secret_value()
            ):
                return old
            self.mutable(request.id)
            node = Node(
                id=info.id,
                name=request.name or info.name,
                endpoint=request.endpoint,
                identity=info,
                ssh=request.ssh,
                checked_at=time.time(),
            )
            self.nodes[info.id] = node
            self.credentials[info.id] = request.api_key.get_secret_value()
            self.save()
            await self.emit("node.registered", node.model_dump())
            return node

    async def update(self, node_id, request):
        async with self.lock:
            self.mutable(node_id)
            old = self.nodes.get(node_id)
            if old is None:
                fail("node_not_found", "Unknown node ID.", 404)
            endpoint = request.endpoint or old.endpoint
            credential = (
                request.api_key.get_secret_value()
                if request.api_key
                else self.credentials[node_id]
            )
            try:
                info = await self.probe(endpoint, credential)
            except Exception:
                fail(
                    "identity_unreachable",
                    "Authenticated worker identity verification failed.",
                    502,
                    retryable=True,
                )
            if info.id != node_id or info.physical_id != old.identity.physical_id:
                fail(
                    "identity_mismatch",
                    "Address updates must preserve device identity.",
                    422,
                )
            if any(
                n.id != node_id and n.endpoint == endpoint for n in self.nodes.values()
            ):
                fail("duplicate_endpoint", "Endpoint belongs to another registration.")
            ssh = request.ssh or old.ssh
            node = old.model_copy(
                update={
                    "endpoint": endpoint,
                    "name": request.name or old.name,
                    "identity": info,
                    "reachable": True,
                    "stale": False,
                    "checked_at": time.time(),
                    "error": None,
                    "ssh": ssh,
                }
            )
            self.nodes[node_id], self.credentials[node_id] = node, credential
            self.save()
            await self.emit("node.updated", node.model_dump())
            return node

    async def bootstrap(self):
        coordinator = runtime.distributed_coordinator
        if self.bootstrapped or coordinator is None:
            return
        # A configured CLI group is approved by its configuration file. Import
        # verified identities without replacing that live coordinator.
        nodes, credentials = {}, {}
        for worker in coordinator.config.workers:
            credential = worker.api_key or os.environ.get(
                coordinator.config.api_key_env, ""
            )
            info = await self.probe(worker.url.rstrip("/"), credential)
            if info.id in nodes:
                fail(
                    "duplicate_device",
                    "Configured workers must have distinct stable node IDs.",
                )
            nodes[info.id] = Node(
                id=info.id,
                name=info.name,
                endpoint=worker.url.rstrip("/"),
                ssh=(
                    SSHTransport(
                        destination=worker.ssh,
                        python=worker.python or coordinator.config.python,
                        rdma=worker.rdma,
                    )
                    if worker.ssh
                    else None
                ),
                identity=info,
                checked_at=time.time(),
            )
            credentials[info.id] = credential
        self.nodes.update(nodes)
        self.credentials.update(credentials)
        self.selected = list(nodes)
        self.running = list(nodes)
        self.backend = next(iter(nodes.values())).identity.worker.get("backend")
        self.state = "ready"
        self.bootstrapped = True
        self.save()

    async def refresh(self, *, force=False):
        async def one(node):
            try:
                info = await self.probe(node.endpoint, self.credentials[node.id])
                if info.id != node.id or info.physical_id != node.identity.physical_id:
                    raise ValueError("Identity changed")
                node.identity, node.reachable, node.stale, node.error = (
                    info,
                    True,
                    False,
                    None,
                )
            except httpx.TimeoutException:
                # Loading can delay telemetry without losing an inference rank.
                node.reachable, node.stale = False, True
                node.error = ClusterError(
                    code="telemetry_timeout",
                    message="Worker identity check timed out; telemetry is stale.",
                    retryable=True,
                )
            except Exception:
                node.reachable, node.stale = False, True
                node.error = ClusterError(
                    code="node_unreachable",
                    message="Authenticated identity check failed.",
                    retryable=True,
                )
            node.checked_at = time.time()

        async with self.lock:
            if not force and time.monotonic() - self.last_refresh < 1:
                return
            if (
                not self.bootstrapped
                and runtime.distributed_coordinator
                and (not self.nodes or runtime.distributed_coordinator.launcher)
            ):
                try:
                    await self.bootstrap()
                except httpx.TimeoutException:
                    pass  # Retry registration when worker telemetry responds.
                except Exception:
                    self.state = "failed"
            self.last_refresh = time.monotonic()
            await asyncio.gather(*(one(node) for node in self.nodes.values()))
            if any(
                self.nodes[n].error and self.nodes[n].error.code == "node_unreachable"
                for n in self.running
            ):
                self.state = "failed"
                if runtime.distributed_coordinator:
                    runtime.distributed_coordinator.failed(
                        RuntimeError("Registered worker lost or identity changed")
                    )
            await self.emit(
                "nodes.telemetry",
                {"nodes": [n.model_dump() for n in self.nodes.values()]},
            )

    def aggregate(self):
        total = used = 0
        physical = set()
        missing, stale = [], []
        for node_id in self.selected:
            node = self.nodes[node_id]
            measurement = node.identity
            if (
                node.stale
                or not node.reachable
                or time.time() - measurement.measured_at > 15
            ):
                stale.append(node_id)
            if (
                measurement.memory.total_bytes is None
                or measurement.memory.used_bytes is None
            ):
                missing.append(node_id)
            if measurement.physical_id in physical:
                continue
            physical.add(measurement.physical_id)
            total += measurement.memory.total_bytes or 0
            used += measurement.memory.used_bytes or 0
        return AggregateMemory(
            total_bytes=total,
            used_bytes=used,
            complete=not missing and not stale,
            stale_node_ids=stale,
            missing_node_ids=missing,
            physical_devices=len(physical),
        )

    def snapshot(self):
        coordinator = runtime.distributed_coordinator
        state = self.state
        if coordinator and coordinator.failure.exists():
            state = "failed"
        primary = None
        for node_id in self.running:
            worker = self.nodes[node_id].identity.worker or {}
            if worker.get("rank") == 0:
                primary = node_id
        return ClusterState(
            state=state,
            coordinator=identity(runtime.distributed_worker),
            inference_coordinator_id=primary,
            selected_members=self.selected,
            running_members=self.running,
            backend=self.backend,
            active_model=(
                coordinator.model[0] if coordinator and coordinator.model else None
            ),
            draft=coordinator.draft if coordinator else None,
            active_requests=coordinator.active if coordinator else 0,
            ownership=(
                "server" if coordinator and coordinator.launcher else "external"
            ),
            fast_sync=(
                getattr(coordinator.launcher.launch, "fast_sync", None)
                if coordinator and coordinator.launcher and self.running
                else None
            ),
            memory=self.aggregate(),
            capabilities=Capabilities(
                speculative_drafters=[
                    kind
                    for kind in ("dflash", "mtp")
                    if all(
                        kind
                        in (self.nodes[n].identity.worker or {}).get(
                            "speculative_drafters", []
                        )
                        for n in self.running
                    )
                ],
                resumable=not self.running
                or all(
                    (self.nodes[n].identity.worker or {}).get("protocol_version") == 2
                    for n in self.running
                ),
            ),
            operation_id=self.current,
        )

    def config(self, node_ids=None):
        prior = runtime.distributed_coordinator
        targets = self.selected if node_ids is None else node_ids
        local_physical = identity(runtime.distributed_worker).physical_id
        for node_id in targets:
            node = self.nodes[node_id]
            if node.ssh is None and node.identity.physical_id != local_physical:
                fail(
                    "ssh_transport_required",
                    f"Node {node_id} needs an existing SSH destination for weight delivery.",
                    422,
                    action="PATCH the registration with ssh.destination and optional ssh.python; configure SSH access outside MLX-VLM.",
                )
        workers = tuple(
            Worker(
                self.nodes[n].endpoint,
                ssh=self.nodes[n].ssh.destination if self.nodes[n].ssh else None,
                python=self.nodes[n].ssh.python if self.nodes[n].ssh else None,
                node_id=n,
                model_dir=self.nodes[n].identity.model_dir,
                api_key=self.credentials[n],
                physical_id=self.nodes[n].identity.physical_id,
                rdma=(
                    self.nodes[n].ssh.rdma.model_dump()
                    if self.nodes[n].ssh and self.nodes[n].ssh.rdma
                    else None
                ),
            )
            for n in targets
        )
        if prior:
            return replace(prior.config, workers=workers)
        return ClusterConfig(
            workers=workers,
            model_dir=self.root / "models",
            shared_lock=self.root / "group.lock",
            python=os.sys.executable,
            compaction_key_file=(
                Path(os.environ["MLX_VLM_COMPACTION_KEY_FILE"])
                if os.environ.get("MLX_VLM_COMPACTION_KEY_FILE")
                else None
            ),
        )

    async def check_model(self, model, revision):
        if not model:
            return
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import RemoteEntryNotFoundError

        from ..utils import load_config

        def metadata():
            local = Path(model).expanduser()
            if local.is_dir():
                if revision:
                    fail(
                        "invalid_revision",
                        "Revision applies to Hub model IDs only.",
                        422,
                    )
                return load_config(local)
            if model.startswith(("/", "~", ".")):
                fail("model_not_found", "Local model directory does not exist.", 422)
            try:
                path = hf_hub_download(model, "config.json", revision=revision)
            except RemoteEntryNotFoundError:
                path = hf_hub_download(model, "params.json", revision=revision)
            return json.loads(Path(path).read_text())

        validate_model(await asyncio.to_thread(metadata), len(self.selected))

    async def select(self, members):
        async with self.lock:
            if (
                members.node_ids == self.selected
                and members.backend == self.backend
                and not members.model
            ):
                return self.snapshot()
            self.mutable()
            unknown = set(members.node_ids) - self.nodes.keys()
            if unknown:
                fail(
                    "unknown_members",
                    f"Register these nodes first: {sorted(unknown)}",
                    422,
                )
            for node_id in members.node_ids:
                if members.backend not in self.nodes[node_id].identity.backends:
                    fail(
                        "backend_unavailable",
                        f"Node {node_id} does not advertise {members.backend}.",
                        422,
                    )
            old, backend = self.selected, self.backend
            self.selected, self.backend = members.node_ids, members.backend
            try:
                await self.check_model(members.model, members.revision)
            except BaseException:
                self.selected, self.backend = old, backend
                raise
            self.save()
            await self.emit(
                "cluster.members", {"node_ids": self.selected, "backend": self.backend}
            )
            return self.snapshot()

    async def begin(self, kind, body, key):
        async with self.lock:
            signature = json.dumps([kind, body.model_dump()], sort_keys=True)
            if key and key in self.idempotency:
                previous = self.idempotency[key]
                if previous[0] != signature:
                    fail(
                        "idempotency_conflict",
                        "This key was used for a different operation.",
                    )
                return self.operations[previous[1]]
            self.idle()
            operation = Operation(
                id=uuid.uuid4().hex,
                kind=kind,
                created_at=time.time(),
                updated_at=time.time(),
            )
            self.current = operation.id
            self.operations[operation.id] = operation
            if kind == "prepare_model":
                self.transfer_stops[operation.id] = threading.Event()
            if key:
                self.idempotency[key] = [signature, operation.id]
            self.save()
            task = asyncio.create_task(self.execute(operation, body))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            return operation

    async def phase(self, operation, phase, *, progress=None):
        if progress is not None:
            operation.result = {"progress": progress}
        operation.phase, operation.updated_at = phase, time.time()
        self.save()
        await self.emit("operation.updated", operation.model_dump())

    async def execute(self, operation, body):
        try:
            if operation.kind == "start":
                await self.start(operation, body)
            elif operation.kind == "prepare_model":
                await self.prepare_model(operation, body)
            else:
                await self.stop(operation, body)
            operation.status = "succeeded"
            if operation.result is None:
                operation.result = {"state": self.state, "members": self.running}
            await self.phase(operation, "completed")
        except BaseException as error:
            operation.status = "failed"
            if isinstance(error, HTTPException) and isinstance(error.detail, dict):
                operation.error = ClusterError.model_validate(error.detail)
            else:
                operation.error = ClusterError(
                    code=(
                        "preparation_cancelled"
                        if isinstance(error, InterruptedError)
                        else "transition_failed"
                    ),
                    message=str(error) or type(error).__name__,
                    retryable=True,
                )
            # Each transition owns its rollback. An operation rejection is not
            # evidence that the running inference group failed.
            await self.phase(operation, "failed")
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            self.transfer_stops.pop(operation.id, None)

    async def prepare_model(self, operation, body):
        targets = body.node_ids if body.node_ids is not None else self.selected
        if not targets:
            fail(
                "members_required", "Select members or supply registered node_ids.", 422
            )
        unknown = set(targets) - self.nodes.keys()
        if unknown:
            fail(
                "unknown_members", f"Register these nodes first: {sorted(unknown)}", 422
            )
        await self.refresh(force=True)
        if any(not self.nodes[n].reachable for n in targets):
            fail(
                "node_unreachable",
                "All delivery targets must authenticate successfully.",
                502,
                retryable=True,
            )
        # Independent immutable delivery cache; never mutate the active model or
        # acquire an inference lease just to copy a checkpoint.
        config = self.config(targets)
        from .checkpoint_transfer import TransferPool

        if self.transfers is None:
            self.transfers = TransferPool()
        coordinator = runtime.distributed_coordinator
        if coordinator:
            coordinator.stager.transfers = self.transfers
        stager = ModelStager(config, transfers=self.transfers)
        prepared = await self.stage_model(
            operation,
            stager,
            body.model,
            body.revision,
            validate_sharding=False,
            transport=body.transport,
            stop=self.transfer_stops[operation.id],
        )
        operation.result = {
            "model": body.model,
            "revision": body.revision,
            "snapshot": Path(prepared).name,
            "nodes": [
                {
                    "id": w.node_id,
                    "path": stager.worker_paths[(prepared, w.url)],
                    "transport": stager.worker_transfers.get(w.url, {}).get(
                        "transport", "ssh_rsync" if w.ssh else "local_filesystem"
                    ),
                    "verified": True,
                    **{
                        key: value
                        for key, value in stager.worker_transfers.get(w.url, {}).items()
                        if key in {"rdma_validated", "fallback_reason"}
                    },
                }
                for w in config.workers
            ],
        }

    async def stage_model(self, operation, stager, model, revision, **options):
        loop = asyncio.get_running_loop()

        async def progress(phase, data):
            operation.result = {"progress": data}
            await self.phase(operation, phase)

        def report(phase, **data):
            asyncio.run_coroutine_threadsafe(progress(phase, data), loop).result()

        return await self.stage(stager, model, revision, progress=report, **options)

    @staticmethod
    async def stage(stager, model, revision, **options):
        # Cancellation of an asyncio waiter cannot stop rsync or its thread.
        # Keep the event loop alive for progress callbacks until staging exits.
        task = asyncio.create_task(
            asyncio.to_thread(stager.prepare, model, revision, **options)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            finally:
                raise

    async def load_model(self, operation, coordinator, body, staged):
        """Observe worker measurements while the existing relay loads the model."""
        started = time.monotonic()
        task = asyncio.create_task(
            coordinator.relay(
                "/v1/responses/input_tokens",
                {
                    "model": body.model,
                    "input": ".",
                    "distributed_revision": body.revision,
                },
            )
        )
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=1) as client:
                while True:
                    checks = await asyncio.gather(
                        *[
                            client.get(
                                w.url.rstrip("/") + "/health", headers=w.headers()
                            )
                            for w in coordinator.config.workers
                        ],
                        return_exceptions=True,
                    )
                    nodes = []
                    for rank, (worker, check) in enumerate(
                        zip(coordinator.config.workers, checks)
                    ):
                        node = {
                            "node_id": worker.node_id,
                            "rank": rank,
                            "stage": "unavailable",
                        }
                        if isinstance(check, httpx.Response) and check.is_success:
                            data = check.json()
                            info = data.get("distributed_worker") or {}
                            expected = coordinator.stager.worker_paths.get(
                                (staged, worker.url), staged
                            )
                            measured = (
                                data.get("model_load_progress")
                                or info.get("load_progress")
                                or {}
                            )
                            node["stage"] = "waiting_for_load"
                            if measured.get("model") == expected:
                                node.update(measured)
                                if node["stage"] == "ready":
                                    node["stage"] = "initializing_generator"
                            if data.get("loaded_model") == expected:
                                node["stage"] = "ready"
                        nodes.append(node)
                    await self.phase(
                        operation,
                        "loading_model",
                        progress={
                            "elapsed_seconds": time.monotonic() - started,
                            "total_workers": len(nodes),
                            "workers_ready": sum(n["stage"] == "ready" for n in nodes),
                            "nodes": nodes,
                        },
                    )
                    if task.done():
                        return await task
                    await asyncio.wait({task}, timeout=0.5)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def start(self, operation, body):
        await self.refresh()
        if self.running:
            fail(
                "already_running",
                "Stop the existing group before starting another group.",
            )
        if len(self.selected) < 2:
            fail("members_required", "Select at least two registered workers.", 422)
        previous_state = self.state
        launcher = None
        try:
            self.state = "starting"
            await self.phase(operation, "validating")
            await self.check_model(body.model, body.revision)
            config = self.config()
            old = runtime.distributed_coordinator
            if body.mode == "attach":
                config = replace(config, launch=None)
            elif config.launch is None:
                fail(
                    "launcher_not_configured",
                    "SSH launch mode requires a configured launch section.",
                    422,
                )
            if old and (old.active or old.waiters):
                fail(
                    "group_busy",
                    "Inference must drain before replacing the group.",
                    retryable=True,
                )
            coordinator = Coordinator(config)
            coordinator.draft = body.draft
            coordinator.draft_target = (
                (body.model, body.revision) if body.draft else None
            )
            coordinator.stager.transfers = self.transfers
            if config.launch:
                if config.launch.backend != self.backend:
                    fail(
                        "launch_reconfiguration_required",
                        "Selected backend differs from the configured SSH launcher topology.",
                        action="Update the launcher configuration while stopped, or attach a compatible external group.",
                    )
                configured = {w.url for w in old.config.workers}
                if {w.url for w in config.workers} != configured:
                    fail(
                        "launch_reconfiguration_required",
                        "Selected members differ from the SSH launcher hostfile.",
                        action="Update the launcher configuration while stopped, or attach application-managed workers.",
                    )
                from .distributed_launch import WorkerLauncher

                coordinator.launcher = launcher = WorkerLauncher(coordinator)

                async def launching(**progress):
                    await self.phase(operation, "launching", progress=progress)

                # Keep the failure marker until fresh process identities are checked
                # below; startup still refuses occupied endpoints or another owner.
                await launcher.start(
                    recover=True, progress=launching, fast_sync=body.fast_sync
                )
            else:
                for rank, node_id in enumerate(self.selected):
                    info = self.nodes[node_id].identity.worker
                    if (
                        not info
                        or info.get("world_size") != len(self.selected)
                        or info.get("backend") != self.backend
                        or info.get("rank") != rank
                    ):
                        operation.result = {
                            "required_action": "start_external_group",
                            "backend": self.backend,
                            "world_size": len(self.selected),
                            "members": [
                                {"id": n, "rank": i, "endpoint": self.nodes[n].endpoint}
                                for i, n in enumerate(self.selected)
                            ],
                        }
                        fail(
                            "external_reconfiguration_required",
                            "External workers must join the selected collective group before attach.",
                            action="Launch the returned rank plan in each worker's own Python environment, then retry start.",
                        )
                group_ids = {
                    (self.nodes[n].identity.worker or {}).get("group_id")
                    for n in self.selected
                }
                if not all(group_ids) or len(group_ids) != 1:
                    fail(
                        "group_identity_mismatch",
                        "External ranks must present the same nonempty MLX_VLM_GROUP_ID.",
                        action="Launch all selected workers together with a fresh shared group ID.",
                    )
            await self.phase(operation, "attaching")
            async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
                await coordinator._health(client)
            if coordinator.failure.exists():
                # An old failure cannot be cleared by health alone: require fresh process identities.
                old_instances = json.loads(coordinator.failure.read_text()).get(
                    "instances"
                ) or getattr(old, "group_instances", None)
                instances = coordinator.group_instances
                if (
                    not old_instances
                    or not instances
                    or any(not value or value in old_instances for value in instances)
                ):
                    fail(
                        "restart_required",
                        "Failed workers must be restarted before recovery.",
                        action="Restart the owned/external group and retry start.",
                    )
                coordinator.failure.unlink()
            if body.model:
                await self.phase(operation, "preparing_model")
                staged = await self.stage_model(
                    operation, coordinator.stager, body.model, body.revision
                )
                if body.draft is not None:
                    from ..speculative.drafters import resolve_drafter_kind

                    draft = body.draft
                    if (draft.model, draft.revision) == (body.model, body.revision):
                        draft_path = staged
                    else:
                        await self.phase(operation, "preparing_draft")
                        draft_path = await self.stage_model(
                            operation,
                            coordinator.stager,
                            draft.model,
                            draft.revision,
                            validate_sharding=False,
                        )
                    kind = resolve_drafter_kind(Path(draft_path), draft.kind)
                    if kind not in {"dflash", "mtp"}:
                        fail(
                            "unsupported_draft",
                            "Distributed drafting supports DFlash/DSpark and MTP.",
                            422,
                        )
                    coordinator.draft = draft.model_copy(update={"kind": kind})
                    coordinator.draft_path = draft_path
                _, status = await self.load_model(operation, coordinator, body, staged)
                if status >= 400:
                    fail(
                        "model_load_failed",
                        "Worker model load failed; inspect authenticated worker health.",
                        502,
                    )
        except BaseException:
            # Only this attempt's resources are eligible for cleanup. Existing
            # launchers (including CLI-owned groups) must remain untouched.
            try:
                if launcher:
                    await launcher.close()
            finally:
                if launcher or self.state == "failed":
                    self.state = "failed"
                else:
                    self.state = previous_state
            raise
        runtime.distributed_coordinator = coordinator
        self.running = list(self.selected)
        self.state = "ready"
        operation.result = {
            **(operation.result or {}),
            "state": self.state,
            "members": self.running,
        }
        self.save()
        await self.emit("cluster.ready", self.snapshot().model_dump())

    async def stop(self, operation, body):
        coordinator = runtime.distributed_coordinator
        previous_state = self.state
        was_draining = coordinator.draining if coordinator else False
        destructive = False
        try:
            self.state = "draining"
            await self.phase(operation, "draining")
            if coordinator:
                coordinator.draining = True
                try:

                    async def drain():
                        async with coordinator.condition:
                            coordinator.condition.notify_all()
                            while coordinator.active or coordinator.waiters:
                                await coordinator.condition.wait()

                    await asyncio.wait_for(drain(), body.drain_timeout_seconds)
                except asyncio.TimeoutError:
                    fail(
                        "drain_timeout",
                        "Active inference has not drained; the group was not stopped.",
                        action="Finish/cancel active requests and retry stop.",
                        retryable=True,
                    )
            if coordinator and coordinator.continuations:
                if not body.discard_continuations:
                    fail(
                        "retained_continuations",
                        "Paused or completed continuation checkpoints remain.",
                        action="Retry stop with discard_continuations=true, or resume the retained sessions.",
                    )
                destructive = True
                if coordinator.failure.exists():
                    # A broken collective cannot acknowledge cache disposal. Keep
                    # its failure marker and require fresh processes on restart.
                    # Owned processes are stopped below; external processes remain
                    # their owner's responsibility, explicitly reported in result.
                    coordinator.continuations.clear()
                    operation.result = {
                        "state": "stopped",
                        "members": [],
                        "external_restart_required": coordinator.launcher is None,
                        "collective_disposal_acknowledged": False,
                    }
                else:
                    async with httpx.AsyncClient(
                        trust_env=False,
                        timeout=65,
                        headers={
                            "Authorization": "Bearer "
                            + os.environ.get(coordinator.config.api_key_env, "")
                        },
                    ) as client:
                        await coordinator.expire_continuations(
                            client, all_sessions=True
                        )
            launcher = coordinator.launcher if coordinator else None
            if launcher:
                await self.phase(operation, "stopping_owned_workers")
                destructive = True
                await launcher.close()
                coordinator.launcher = None
            # Detaching an external group never sends a process-stop command.
            self.running, self.state = [], "stopped"
            self.bootstrapped = True
            self.save()
            await self.emit("cluster.stopped", {"external_processes_stopped": False})
        except BaseException:
            if (
                destructive
                or self.state == "failed"
                or (coordinator and coordinator.failure.exists())
            ):
                # Disposal or process shutdown may have partially completed;
                # admitting inference again would be unsafe.
                self.state = "failed"
            else:
                self.state = previous_state
                if coordinator:
                    coordinator.draining = was_draining
            raise

    async def close(self):
        for stop in self.transfer_stops.values():
            stop.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        coordinator = runtime.distributed_coordinator
        if coordinator and coordinator.launcher:
            await coordinator.launcher.close()
        if self.transfers:
            await asyncio.to_thread(self.transfers.close)


def manager():
    if runtime.cluster_manager is None:
        runtime.cluster_manager = ClusterManager()
    return runtime.cluster_manager


def router(authenticate):
    from fastapi import Depends

    api = APIRouter(
        dependencies=[Depends(authenticate)],
        tags=["Cluster"],
        responses={
            409: {
                "description": "Structured conflict (detail: ClusterError)",
                "model": ClusterErrorResponse,
            }
        },
    )

    @api.get("/internal/distributed/identity", response_model=NodeIdentity)
    async def worker_identity():
        return await asyncio.to_thread(identity, runtime.distributed_worker)

    @api.get("/v1/cluster", response_model=ClusterState)
    async def get_cluster():
        await manager().refresh()
        return manager().snapshot()

    @api.get("/v1/cluster/nodes", response_model=list[Node])
    async def get_nodes():
        await manager().refresh()
        return list(manager().nodes.values())

    @api.post("/v1/cluster/nodes", response_model=Node)
    async def register_node(body: RegisterNode):
        return await manager().register(body)

    @api.patch("/v1/cluster/nodes/{node_id}", response_model=Node)
    async def update_node(node_id: str, body: UpdateNode):
        return await manager().update(node_id, body)

    @api.delete("/v1/cluster/nodes/{node_id}", status_code=204)
    async def remove_node(node_id: str):
        control = manager()
        async with control.lock:
            if node_id not in control.nodes:
                return
            control.mutable(node_id)
            control.nodes.pop(node_id, None)
            control.credentials.pop(node_id, None)
            control.selected = [n for n in control.selected if n != node_id]
            control.save()
            await control.emit("node.removed", {"id": node_id})

    @api.put("/v1/cluster/members", response_model=ClusterState)
    async def select_members(body: Members):
        return await manager().select(body)

    @api.post("/v1/cluster/start", status_code=202, response_model=Operation)
    async def start_group(
        body: StartGroup = StartGroup(),
        idempotency_key: str | None = Header(default=None, max_length=256),
    ):
        return await manager().begin("start", body, idempotency_key)

    @api.post("/v1/cluster/stop", status_code=202, response_model=Operation)
    async def stop_group(
        body: StopGroup = StopGroup(),
        idempotency_key: str | None = Header(default=None, max_length=256),
    ):
        return await manager().begin("stop", body, idempotency_key)

    @api.post("/v1/cluster/models/prepare", status_code=202, response_model=Operation)
    async def prepare_model(
        body: PrepareModel,
        idempotency_key: str | None = Header(default=None, max_length=256),
    ):
        return await manager().begin("prepare_model", body, idempotency_key)

    @api.get("/v1/cluster/operations/{operation_id}", response_model=Operation)
    async def operation_status(operation_id: str):
        operation = manager().operations.get(operation_id)
        if operation is None:
            fail("operation_not_found", "Unknown operation ID.", 404)
        return operation

    @api.post(
        "/v1/cluster/operations/{operation_id}/cancel",
        status_code=202,
        response_model=Operation,
    )
    async def cancel_preparation(operation_id: str):
        control = manager()
        async with control.lock:
            operation = control.operations.get(operation_id)
            if operation is None:
                fail("operation_not_found", "Unknown operation ID.", 404)
            if operation.status != "running":
                return operation
            if operation.phase == "publishing":
                fail(
                    "cancellation_too_late",
                    "Verified snapshot publication has already started.",
                )
            stop = control.transfer_stops.get(operation_id)
            if stop is None:
                fail(
                    "cancellation_unavailable",
                    "This operation cannot be cancelled through the transfer API.",
                )
            stop.set()
            await control.phase(operation, "cancelling")
            return operation

    @api.get(
        "/v1/cluster/events",
        response_model=None,
        responses={
            200: {
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
                "description": "SSE ClusterEvent records; Last-Event-ID replay or cluster.snapshot reset.",
            }
        },
    )
    async def events(
        last_event_id: int | None = Header(default=None),
        interval_seconds: float = Query(default=2, ge=1, le=60),
    ):
        control = manager()

        async def stream():
            cursor = last_event_id
            while True:
                if (
                    cursor is None
                    or cursor > control.event_id
                    or (control.events and cursor < control.events[0].id - 1)
                ):
                    event = ClusterEvent(
                        id=control.event_id,
                        type="cluster.snapshot",
                        timestamp=time.time(),
                        data=control.snapshot().model_dump(),
                    )
                    cursor = event.id
                    yield f"id: {event.id}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n"
                for event in list(control.events):
                    if event.id > cursor:
                        cursor = event.id
                        yield f"id: {event.id}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n"
                await asyncio.sleep(interval_seconds)
                await control.refresh()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return api
