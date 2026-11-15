# Copyright © 2026 Neywa Labs Inc.

"""Opt-in request routing to a dedicated group of distributed HTTP workers.

The coordinator never joins the collective group. Its ordinary model loader and
GPU thread remain independent of the workers (including the worker on this Mac).
"""

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import secrets
import time
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from .checkpoint_store import MANIFEST
from .cluster_types import RDMAPair, SSHTransport, model_directory, worker_origin
from .distributed_launch import LaunchConfig

logger = logging.getLogger("mlx_vlm.server.distributed")

ROUTES = {
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/responses",
    "/responses",
    "/v1/responses/input_tokens",
    "/responses/input_tokens",
    "/v1/responses/compact",
    "/responses/compact",
}
CANCEL_ROUTE = "/internal/distributed/cancel"
REQUEST_HEADER = "X-MLX-Distributed-Request"
MODEL_SUFFIXES = {
    ".json",
    ".jsonl",
    ".safetensors",
    ".py",
    ".model",
    ".tiktoken",
    ".txt",
    ".jinja",
    ".jinja2",
    ".vocab",
    ".merges",
}


def _digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Worker:
    url: str
    ssh: str | None = None
    node_id: str | None = None
    model_dir: str | None = None
    python: str | None = None
    api_key: str | None = field(default=None, repr=False)
    physical_id: str | None = None
    rdma: dict | None = None

    def __post_init__(self):
        object.__setattr__(self, "url", worker_origin(self.url))
        if self.rdma is not None:
            RDMAPair.model_validate(self.rdma)
        if self.ssh is not None:
            SSHTransport(destination=self.ssh)
        if self.model_dir is not None:
            model_directory(self.model_dir)

    def headers(self):
        return {"Authorization": "Bearer " + self.api_key} if self.api_key else {}


@dataclass(frozen=True)
class ClusterConfig:
    workers: tuple[Worker, ...]
    model_dir: Path
    shared_lock: Path
    python: str = "python3"
    ssh_options: tuple[str, ...] = ()
    api_key_env: str = "MLX_VLM_DISTRIBUTED_API_KEY"
    timeout: float = 7200
    compaction_key_file: Path | None = None
    launch: LaunchConfig | None = None

    @classmethod
    def read(cls, filename):
        data = json.loads(Path(filename).read_text())
        workers = tuple(Worker(**item) for item in data["workers"])
        if len(workers) < 2 or len({w.url for w in workers}) != len(workers):
            raise ValueError("Configure at least two distinct worker URLs")
        root = Path(data["model_dir"]).expanduser()
        model_directory(str(root))
        lock = Path(data["shared_lock"]).expanduser()
        if not lock.is_absolute():
            raise ValueError("shared_lock must be an absolute path")
        timeout = float(data.get("timeout", 7200))
        if not 0 < timeout < float("inf"):
            raise ValueError("timeout must be a positive finite number")
        options = data.get("ssh_options", [])
        if not isinstance(options, list) or not all(
            isinstance(x, str) for x in options
        ):
            raise ValueError("ssh_options must be an array of arguments")
        key = data.get("compaction_key_file")
        launch = (
            LaunchConfig.read(data["launch"], Path(filename).resolve().parent)
            if "launch" in data
            else None
        )
        if launch:
            if not Path(data.get("python", "python3")).is_absolute():
                raise ValueError("Automatic launch requires an absolute python path")
            launch.validate_hosts(workers)
        return cls(
            workers,
            root,
            lock,
            data.get("python", "python3"),
            tuple(options),
            data.get("api_key_env", "MLX_VLM_DISTRIBUTED_API_KEY"),
            timeout,
            Path(key).expanduser() if key else None,
            launch,
        )


class ModelStager:
    def __init__(self, config, transfers=None):
        self.config = config
        self.sources = {}
        self.worker_paths = {}
        self.worker_transfers = {}
        self.transfers = transfers

    def _resolve(self, model, revision):
        from ..utils import get_model_path

        local = Path(model).expanduser()
        if local.is_dir():
            if revision:
                raise ValueError("distributed_revision applies to Hub model IDs only")
            return local.resolve()
        if model.startswith(("/", "~", ".")):
            raise ValueError(f"Local model directory does not exist: {model}")
        # Native snapshot_download reuses cached blobs and downloads missing files.
        # A local-files-only snapshot lookup cannot establish that every tokenizer
        # or weight file is present. Only this coordinator contacts the Hub.
        key = (model, revision)
        if key not in self.sources:
            self.sources[key] = get_model_path(model, revision=revision)
        return self.sources[key]

    def prepare(
        self,
        model,
        revision=None,
        *,
        validate_sharding=True,
        progress=None,
        transport="ssh_rsync",
        stop=None,
    ):
        # Management preparation and inference can target the same immutable
        # snapshot. Serialize staging even when their coordinator caches differ.
        from .checkpoint_store import cancelled
        from .checkpoint_transfer import TransferProgress

        staging_lock = Path(str(self.config.shared_lock) + ".staging")
        staging_lock.parent.mkdir(parents=True, exist_ok=True)
        with staging_lock.open("a+") as lock:
            while True:
                cancelled(stop)
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.1)
            with TransferProgress() as terminal:
                return self._prepare(
                    model,
                    revision,
                    validate_sharding=validate_sharding,
                    progress=progress,
                    transport=transport,
                    stop=stop,
                    terminal=terminal,
                )

    def _prepare(
        self, model, revision, *, validate_sharding, progress, transport, stop, terminal
    ):
        from ..utils import get_model_and_args, load_config
        from .checkpoint_store import prepare_snapshot
        from .checkpoint_transfer import TransferPool

        if transport not in {"auto", "jaccl_rdma", "ssh_rsync"}:
            raise ValueError("Unsupported checkpoint transport")
        self.worker_transfers.clear()

        def report(phase, *, target="Source", **data):
            terminal.update(target, phase, data)
            if progress:
                progress(phase, **data)

        report("resolving_model")
        source = self._resolve(model, revision)
        config = load_config(source)
        if config.get("model_file"):
            raise ValueError(
                "Custom model_file checkpoints are not supported in distributed mode"
            )
        if validate_sharding:
            architecture, _ = get_model_and_args(config)
            if not callable(getattr(architecture.Model, "shard", None)):
                raise ValueError("This model has no distributed shard implementation")
            from .cluster import validate_model

            validate_model(config, len(self.config.workers))
        names = sorted(
            p.relative_to(source).as_posix()
            for p in source.rglob("*")
            if p.is_file()
            and p.suffix in MODEL_SUFFIXES
            and not any(part.startswith(".") for part in p.relative_to(source).parts)
        )
        if not {"config.json", "params.json"}.intersection(names) or not any(
            n.endswith(".safetensors") for n in names
        ):
            raise ValueError(
                "Model must contain config.json or params.json and safetensors weights"
            )
        for name in names:
            if "\n" in name or "\r" in name:
                raise ValueError("Model filenames cannot contain newlines")
            if name.endswith(".safetensors.index.json"):
                missing = set(
                    json.loads((source / name).read_text())
                    .get("weight_map", {})
                    .values()
                ) - set(names)
                if missing:
                    raise ValueError(
                        f"Incomplete checkpoint: missing {sorted(missing)}"
                    )
        last = [0.0]

        def verifying(**data):
            now = time.monotonic()
            if now - last[0] >= 0.2:
                last[0] = now
                report("verifying_source", **data)

        pool = self.transfers or TransferPool()
        try:
            first_result = None
            first = next((w for w in self.config.workers if w.rdma), None)
            if first is not None and transport != "ssh_rsync":
                final, manifest, first_result = pool.prepare_and_deliver(
                    self.config,
                    first,
                    source,
                    names,
                    transport,
                    lambda phase, **data: report(
                        phase,
                        target=first.node_id or first.url,
                        node_id=first.node_id,
                        **data,
                    ),
                    verifying,
                    stop,
                )
            else:
                final, manifest = prepare_snapshot(
                    source, names, self.config.model_dir, stop=stop, progress=verifying
                )
            total = sum(record["size"] for record in manifest["files"].values())
            terminal.update(
                "Source", "source_ready", dict(total_bytes=total, bytes_verified=total)
            )
            for worker in self.config.workers:

                def worker_report(phase, **data):
                    report(
                        phase,
                        target=worker.node_id or worker.url,
                        node_id=worker.node_id,
                        snapshot=final.name,
                        **data,
                    )

                result = first_result if worker is first else None
                if result is None:
                    result = pool.deliver(
                        self.config,
                        worker,
                        final,
                        manifest,
                        transport,
                        worker_report,
                        stop,
                    )
                self.worker_paths[(str(final), worker.url)] = result["path"]
                self.worker_transfers[worker.url] = result
            return str(final)
        finally:
            if self.transfers is None:
                pool.close()


@contextmanager
def cluster_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HTTPException(409, "The distributed cluster is busy") from exc
        yield


def _semantic(value, key_file):
    if isinstance(value, list):
        return [_semantic(item, key_file) for item in value]
    if isinstance(value, dict):
        if value.get("type") == "compaction":
            from cryptography.fernet import Fernet

            from .compaction import CAPSULE_PREFIX

            encrypted = value.get("encrypted_content", "")
            if key_file is None or not encrypted.startswith(CAPSULE_PREFIX):
                raise ValueError(
                    "Compaction requires the same key on coordinator and workers"
                )
            payload = Fernet(key_file.read_bytes().strip()).decrypt(
                encrypted[len(CAPSULE_PREFIX) :].encode()
            )
            return {
                "type": "compaction",
                "context": _semantic(json.loads(payload), key_file),
            }
        ignored = {
            "id",
            "call_id",
            "response_id",
            "item_id",
            "created",
            "created_at",
            "metrics",
            "timings",
            "continuation",
        }
        return {k: _semantic(v, key_file) for k, v in value.items() if k not in ignored}
    return value


class DistributedStreamingResponse(StreamingResponse):
    def __init__(self, content, cleanup):
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        import anyio

        try:
            await super().__call__(scope, receive, send)
        finally:
            # A disconnect can happen while sending headers, before iteration
            # starts. Async-generator finalizers alone cannot cover that case.
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
                await self.cleanup()


class Coordinator:
    def __init__(self, config):
        self.config = config
        self.stager = ModelStager(config)
        self.condition = asyncio.Condition()
        self.waiters = deque()
        self.active = 0
        self.model = None
        self.primary = 0
        self.seed = secrets.randbelow(2**31)
        self.failure = config.model_dir / ".cluster-failed.json"
        self.launch_id = None
        self.launcher = None
        self.draining = False
        self.group_instances = None
        self.continuations = {}
        self.session_protocol = False
        self.resume_lock = asyncio.Lock()

    def failed(self, error):
        self.failure.parent.mkdir(parents=True, exist_ok=True)
        self.failure.write_text(
            json.dumps(
                {
                    "error": type(error).__name__,
                    "message": str(error),
                    "instances": self.group_instances,
                }
            )
        )
        logger.error("Distributed group requires recovery: %s", error)

    async def _health(self, client):
        try:
            checks = await asyncio.gather(
                *[
                    client.get(w.url.rstrip("/") + "/health", headers=w.headers())
                    for w in self.config.workers
                ]
            )
            ranks, signatures = [], []
            for check in checks:
                check.raise_for_status()
                info = check.json().get("distributed_worker") or {}
                if self.launch_id and (
                    info.get("launch_id") != self.launch_id
                    or info.get("backend") != self.config.launch.backend
                ):
                    raise ValueError("Worker does not belong to this server launch")
                if info.get("world_size") != len(checks):
                    raise ValueError(
                        "Worker group size does not match configured cluster"
                    )
                ranks.append(info.get("rank"))
                signatures.append((info.get("backend"), info.get("fingerprint")))
            if sorted(ranks) != list(range(len(checks))):
                raise ValueError(
                    "Configure exactly one endpoint for each distributed rank"
                )
            if not signatures[0][1] or len(set(signatures)) != 1:
                raise ValueError("Distributed worker code or configuration differs")
            self.primary = ranks.index(0)
            self.group_instances = [
                c.json().get("distributed_worker", {}).get("instance_id")
                for c in checks
            ]
            self.session_protocol = all(
                c.json().get("distributed_worker", {}).get("protocol_version") == 2
                for c in checks
            )
        except Exception as error:
            raise HTTPException(
                503, "Distributed worker health/group check failed"
            ) from error

    @asynccontextmanager
    async def _lease(self, model, revision, client):
        """Share a model lease; switch models only after every response drains.

        FIFO tickets keep a waiting model switch from starving behind new rows.
        The process lock spans all overlapping requests, including stream cleanup.
        """
        ticket = object()
        key = (model, revision)
        async with self.condition:
            if self.draining:
                raise HTTPException(
                    409,
                    {
                        "code": "cluster_draining",
                        "message": "Distributed group is draining",
                        "retryable": True,
                    },
                )
            self.waiters.append(ticket)
            acquired = False
            try:
                while self.waiters[0] is not ticket or (
                    self.active and self.model != key
                ):
                    await self.condition.wait()
                if self.draining:
                    raise HTTPException(409, "Distributed group is draining")
                if self.failure.exists():
                    raise HTTPException(
                        503,
                        "Distributed group needs recovery; inspect .cluster-failed.json",
                    )
                if not self.active:
                    lock = cluster_lock(self.config.shared_lock)
                    lock.__enter__()
                    acquired = True
                    await self._health(client)
                    staging = asyncio.create_task(
                        asyncio.to_thread(self.stager.prepare, model, revision)
                    )
                    try:
                        self.model_path = await asyncio.shield(staging)
                    except asyncio.CancelledError:
                        try:
                            await staging
                        finally:
                            raise
                    except Exception as error:
                        raise HTTPException(
                            400, f"Model staging failed: {error}"
                        ) from error
                    self.model = key
                    self.file_lock = lock
                self.active += 1
            except BaseException:
                if acquired:
                    lock.__exit__(None, None, None)
                raise
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()
        try:
            yield self.model_path
        finally:
            async with self.condition:
                self.active -= 1
                if not self.active:
                    self.file_lock.__exit__(None, None, None)
                self.condition.notify_all()

    async def _cancel(self, client, request_id):
        replies = await asyncio.gather(
            *[
                client.post(
                    w.url.rstrip("/") + CANCEL_ROUTE,
                    json={"request_id": request_id},
                    timeout=15,
                    headers=w.headers(),
                )
                for w in self.config.workers
            ],
            return_exceptions=True,
        )
        if any(isinstance(r, BaseException) or r.status_code != 202 for r in replies):
            raise RuntimeError("Could not coordinate cancellation across all ranks")

    async def _cancel_and_drain(self, client, request_id, task):
        try:
            await self._cancel(client, request_id)
            # Do not release the model lease while old GPU work might remain.
            await asyncio.wait_for(asyncio.shield(task), min(60, self.config.timeout))
        except Exception as error:
            self.failed(error)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    def _public_payload(payload, staged, requested):
        if isinstance(payload, dict):
            if payload.get("model") == staged:
                payload["model"] = requested
            response = payload.get("response")
            if isinstance(response, dict) and response.get("model") == staged:
                response["model"] = requested
        return payload

    async def relay(self, path, body, tenant=None):
        import httpx

        if body.get("previous_response_id") or body.get("store") is True:
            raise HTTPException(
                400,
                {
                    "code": "unsupported_feature",
                    "message": "Distributed stored Responses and previous_response_id are not supported",
                    "capability": "stored_responses",
                },
            )
        resumable = bool(body.get("resumable") or body.get("continuation"))
        if resumable and not body.get("stream"):
            raise HTTPException(422, "Distributed continuation requires stream=true")
        checkpoint = body.get("continuation")
        if checkpoint:
            retained = (
                self.continuations.get(checkpoint.get("id"))
                if isinstance(checkpoint, dict)
                else None
            )
            if (
                not retained
                or retained["tenant"] != tenant
                or retained["model"] != body.get("model")
                or retained["path"] != path
            ):
                raise HTTPException(
                    409,
                    "Distributed continuation is unavailable for this tenant/model/endpoint",
                )
            if any(
                record.get("active")
                and record.get("session_id", handle)
                == retained.get("session_id", checkpoint["id"])
                for handle, record in self.continuations.items()
            ):
                raise HTTPException(
                    409,
                    {
                        "code": "continuation_busy",
                        "message": "Close the active reader and wait for its pause before resuming.",
                        "retryable": True,
                    },
                )
        if body.get("adapter_path"):
            raise HTTPException(400, "Distributed adapters are not supported")
        if not isinstance(body.get("model"), str) or not body["model"].strip():
            raise HTTPException(422, "A model identifier is required")
        revision = body.get("distributed_revision")
        if revision is not None and not isinstance(revision, str):
            raise HTTPException(422, "distributed_revision must be a string")
        if (
            path.endswith("/compact") or body.get("context_management")
        ) and self.config.compaction_key_file is None:
            raise HTTPException(
                400, "Configure a shared compaction_key_file before using compaction"
            )
        key = os.environ.get(self.config.api_key_env)
        if not key and not all(w.api_key for w in self.config.workers):
            raise HTTPException(
                503, f"Set worker credentials in {self.config.api_key_env}"
            )
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        if tenant:
            headers["X-Tenant-ID"] = tenant
        request_id = secrets.token_hex(16)
        stack = AsyncExitStack()
        client = await stack.enter_async_context(
            httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout, connect=15),
                trust_env=False,
                headers=headers,
            )
        )
        opening = None
        cancelled = False
        try:
            await self.expire_continuations(client)
            if self.continuations and self.model != (body["model"], revision):
                raise HTTPException(
                    409,
                    {
                        "code": "retained_continuations",
                        "message": "Dispose retained continuations before switching models",
                        "action": "POST /v1/cluster/stop with discard_continuations=true",
                    },
                )
            model_path = await stack.enter_async_context(
                self._lease(body["model"], revision, client)
            )
            if resumable and not self.session_protocol:
                raise HTTPException(
                    422,
                    {
                        "code": "unsupported_feature",
                        "capability": "resumable",
                        "message": "Every worker must support distributed session protocol version 2.",
                    },
                )
            request = {**body, "model": model_path, "distributed": True, "store": False}
            request.pop("distributed_revision", None)
            if request.get("seed") is None:
                # Native batching groups by sampler seed. One shared default allows
                # overlapping default requests to batch; row IDs separate RNG keys.
                request["seed"] = self.seed

            async def open_worker(worker):
                reply = await client.send(
                    client.build_request(
                        "POST",
                        worker.url.rstrip("/") + path,
                        json={
                            **request,
                            "model": self.stager.worker_paths.get(
                                (model_path, worker.url), model_path
                            ),
                        },
                        headers={REQUEST_HEADER: request_id, **worker.headers()},
                    ),
                    stream=True,
                )
                stack.push_async_callback(reply.aclose)
                return reply

            async def open_workers():
                tasks = [
                    asyncio.create_task(open_worker(w)) for w in self.config.workers
                ]
                try:
                    # A failed rank must not wait behind another rank blocked
                    # in a collective until the full generation timeout expires.
                    replies = await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                if cancelled:
                    await asyncio.gather(*(reply.aread() for reply in replies))
                    return replies
                if len({reply.status_code for reply in replies}) != 1:
                    raise RuntimeError("Distributed rank statuses differ")
                return replies

            # Serialize native session claims, not decoding. Each worker must
            # observe replacement readers in the same order.
            if checkpoint:
                async with self.resume_lock:
                    if any(
                        record.get("active")
                        and record.get("session_id", handle)
                        == retained.get("session_id", checkpoint["id"])
                        for handle, record in self.continuations.items()
                    ):
                        raise HTTPException(
                            409,
                            {
                                "code": "continuation_busy",
                                "message": "Another reader resumed this session.",
                                "retryable": True,
                            },
                        )
                    opening = asyncio.create_task(open_workers())
                    replies = await asyncio.shield(opening)
                    if replies[self.primary].is_success:
                        self.continuations[request_id] = {
                            **retained,
                            "session_id": retained.get("session_id", checkpoint["id"]),
                            "active": True,
                            "touched": time.monotonic(),
                        }
            else:
                opening = asyncio.create_task(open_workers())
                replies = await asyncio.shield(opening)
            streaming = bool(body.get("stream")) and replies[self.primary].is_success
            if not streaming:

                async def read_replies():
                    await asyncio.gather(*(reply.aread() for reply in replies))
                    if cancelled:
                        return None
                    payloads = [
                        self._public_payload(
                            reply.json(),
                            self.stager.worker_paths.get(
                                (model_path, worker.url), model_path
                            ),
                            body["model"],
                        )
                        for reply, worker in zip(replies, self.config.workers)
                    ]
                    if replies[0].status_code >= 500:
                        raise RuntimeError(
                            "A distributed worker reported a server error"
                        )
                    if replies[0].is_success:
                        values = [
                            _semantic(p, self.config.compaction_key_file)
                            for p in payloads
                        ]
                        if any(value != values[0] for value in values[1:]):
                            raise RuntimeError("Distributed rank outputs differ")
                    return (
                        self._public_payload(
                            payloads[self.primary], model_path, body["model"]
                        ),
                        replies[self.primary].status_code,
                    )

                reading = asyncio.create_task(read_replies())
                try:
                    result = await asyncio.shield(reading)
                except asyncio.CancelledError:
                    cancelled = True
                    await self._cancel_and_drain(client, request_id, reading)
                    opening = None  # This request has already drained.
                    raise
                await stack.aclose()
                return result

            return self._stream(
                replies,
                client,
                request_id,
                model_path,
                body["model"],
                stack,
                resumable=resumable,
                tenant=tenant,
                path=path,
            )
        except asyncio.CancelledError:
            cancelled = True
            if opening is not None:

                async def drain_opened():
                    replies = await opening
                    await asyncio.gather(*(reply.aread() for reply in replies))

                await self._cancel_and_drain(
                    client, request_id, asyncio.create_task(drain_opened())
                )
            await stack.aclose()
            raise
        except BaseException as error:
            if not isinstance(error, HTTPException):
                self.failed(error)
                if opening is not None:
                    try:
                        await self._cancel(client, request_id)
                    except Exception:
                        logger.exception(
                            "Cancellation after distributed failure failed"
                        )
            await stack.aclose()
            if isinstance(error, HTTPException):
                raise
            raise HTTPException(
                502, "Distributed request failed; group marked for recovery"
            ) from error

    async def session_action(self, client, record_id, action):
        from .distributed_sessions import SESSION_ROUTE

        replies = await asyncio.gather(
            *[
                client.post(
                    worker.url.rstrip("/") + SESSION_ROUTE,
                    json={"id": record_id, "action": action},
                    headers=worker.headers(),
                    timeout=65,
                )
                for worker in self.config.workers
            ],
            return_exceptions=True,
        )
        if any(
            isinstance(reply, BaseException) or not reply.is_success
            for reply in replies
        ):
            error = RuntimeError(
                "Distributed session control failed; restart the group"
            )
            self.failed(error)
            raise error
        if action == "dispose":
            family = self.continuations.get(record_id, {}).get("session_id", record_id)
            self.continuations = {
                handle: record
                for handle, record in self.continuations.items()
                if record.get("session_id", handle) != family
            }

    async def expire_continuations(self, client, *, all_sessions=False):
        async with self.resume_lock:
            families = {}
            for record_id, record in self.continuations.items():
                families.setdefault(record.get("session_id", record_id), []).append(
                    (record_id, record)
                )
            for records in families.values():
                if all_sessions or all(
                    not record.get("active")
                    and time.monotonic() - record["touched"] > record["ttl"]
                    for _, record in records
                ):
                    record_id, _ = max(records, key=lambda pair: pair[1]["touched"])
                    await self.session_action(client, record_id, "dispose")

    def _stream(
        self,
        replies,
        client,
        request_id,
        staged,
        requested,
        stack,
        *,
        resumable=False,
        tenant=None,
        path=None,
    ):
        queue = asyncio.Queue(maxsize=64)
        abandoned = False
        terminal = []
        digests = [hashlib.sha256() for _ in replies]
        counts = [0 for _ in replies]
        ended = [False for _ in replies]

        async def read_rank(rank, reply):
            lines = []

            async def event():
                if not lines:
                    return
                data = "\n".join(
                    line[5:].lstrip(" ") for line in lines if line.startswith("data:")
                )
                name = next(
                    (line[6:].strip() for line in lines if line.startswith("event:")),
                    "",
                )
                if not data:
                    return
                payload = json.loads(data) if data != "[DONE]" else data
                payload = self._public_payload(
                    payload,
                    self.stager.worker_paths.get(
                        (staged, self.config.workers[rank].url), staged
                    ),
                    requested,
                )
                if (
                    not abandoned
                    and isinstance(payload, dict)
                    and (
                        payload.get("error")
                        or payload.get("type") in {"error", "response.failed"}
                    )
                ):
                    raise RuntimeError("A distributed worker emitted a stream error")
                progress = isinstance(payload, dict) and payload.get("type") in {
                    "response.prompt_progress",
                    "response.generation_progress",
                }
                if not progress:
                    canonical = json.dumps(
                        [name, _semantic(payload, self.config.compaction_key_file)],
                        sort_keys=True,
                    )
                    digests[rank].update(canonical.encode() + b"\n")
                    counts[rank] += 1
                if rank == self.primary and resumable:
                    checkpoint = (
                        payload.get("continuation", {})
                        if isinstance(payload, dict)
                        else {}
                    )
                    previous = self.continuations.get(request_id, {})
                    self.continuations[request_id] = {
                        "session_id": previous.get("session_id", request_id),
                        "tenant": tenant,
                        "model": requested,
                        "path": path,
                        "touched": time.monotonic(),
                        "active": True,
                        "ttl": checkpoint.get(
                            "idle_timeout_seconds", previous.get("ttl", 120)
                        ),
                    }
                done = data == "[DONE]" or (
                    isinstance(payload, dict)
                    and payload.get("type")
                    in {"response.completed", "response.incomplete"}
                )
                ended[rank] |= done
                if rank == self.primary and not abandoned:
                    public = self._public_payload(payload, staged, requested)
                    outgoing = (
                        (f"event: {name}\n" if name else "")
                        + "data: "
                        + ("[DONE]" if data == "[DONE]" else json.dumps(public))
                        + "\n\n"
                    )
                    if done:
                        terminal.append(outgoing)
                    else:
                        await queue.put(outgoing)

            async for line in reply.aiter_lines():
                if line == "":
                    await event()
                    lines = []
                else:
                    lines.append(line)
            await event()

        async def pump():
            tasks = [
                asyncio.create_task(read_rank(rank, reply))
                for rank, reply in enumerate(replies)
            ]
            try:
                await asyncio.gather(*tasks)
                if not abandoned:
                    summaries = [
                        (digest.hexdigest(), count)
                        for digest, count in zip(digests, counts)
                    ]
                    if not all(ended) or any(
                        value != summaries[0] for value in summaries[1:]
                    ):
                        raise RuntimeError(
                            "Distributed streams diverged or ended before completion"
                        )
                    for event in terminal:
                        await queue.put(event)
            except Exception as error:
                if abandoned:
                    raise
                if not abandoned:
                    self.failed(error)
                    await queue.put(
                        "event: error\ndata: "
                        + json.dumps(
                            {
                                "error": {
                                    "message": "Distributed stream failed; group marked for recovery",
                                    "type": "distributed_error",
                                }
                            }
                        )
                        + "\n\n"
                    )
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await queue.put(None)

        task = asyncio.create_task(pump())
        cleanup_lock = asyncio.Lock()
        closed = False

        async def cleanup():
            nonlocal abandoned, closed
            import anyio

            # Starlette cancels a task group on disconnect. Shield both the
            # collective cancellation and resource teardown from level cancellation.
            with anyio.CancelScope(shield=True):
                async with cleanup_lock:
                    if closed:
                        return
                    abandoned = True
                    while not queue.empty():
                        queue.get_nowait()
                    try:
                        if not task.done():
                            if resumable and not self.failure.exists():
                                try:
                                    await self.session_action(
                                        client, request_id, "pause"
                                    )
                                finally:
                                    task.cancel()
                                    await asyncio.gather(task, return_exceptions=True)
                            else:
                                await self._cancel_and_drain(client, request_id, task)
                        if resumable and request_id in self.continuations:
                            self.continuations[request_id]["touched"] = time.monotonic()
                            self.continuations[request_id]["active"] = False
                    finally:
                        await stack.aclose()
                        closed = True

        async def iterate():
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
                await task
            finally:
                await cleanup()

        return DistributedStreamingResponse(iterate(), cleanup)


class DistributedWorker:
    def __init__(self, backend, model_dir):
        import mlx.core as mx

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.group = mx.distributed.init(backend=backend, strict=True)
        if self.group.size() < 2:
            raise ValueError("Distributed workers require at least two ranks")
        self.backend = backend
        self.instance_id = secrets.token_hex(16)
        self.load_progress = None
        self._cancel_lock = Lock()
        self._cancelled_requests = set()
        digest = hashlib.sha256()
        package = Path(__file__).parent.parent
        for source in sorted(package.rglob("*.py")):
            digest.update(source.relative_to(package).as_posix().encode())
            digest.update(source.read_bytes())
        digest.update(mx.__version__.encode())
        self.source_hash = digest.hexdigest()

    def remember_cancellations(self, request_ids):
        with self._cancel_lock:
            self._cancelled_requests.update(request_ids)

    def cancelled_requests(self):
        with self._cancel_lock:
            return sorted(self._cancelled_requests)

    def info(self):
        from .runtime import runtime

        settings = {
            name: os.environ.get(name)
            for name in (
                "PREFILL_STEP_SIZE",
                "MLX_VLM_MAX_NUM_SEQS",
                "MLX_VLM_MAX_TOKENS",
                "MLX_VLM_ENABLE_THINKING",
                "MLX_VLM_THINKING_BUDGET",
                "MLX_VLM_THINKING_START_TOKEN",
                "MLX_VLM_THINKING_END_TOKEN",
                "MLX_VLM_MIN_PIXELS",
                "MLX_VLM_MAX_PIXELS",
                "TOP_LOGPROBS_K",
                "MLX_TRUST_REMOTE_CODE",
            )
        }
        key_file = os.environ.get("MLX_VLM_COMPACTION_KEY_FILE")
        if key_file:
            settings["compaction_key_hash"] = _digest_file(key_file)
        signature = json.dumps(
            [self.source_hash, runtime.config.current(), settings], sort_keys=True
        )
        return {
            "protocol_version": 2,
            "inference_validated_at": getattr(self, "inference_validated_at", None),
            "instance_id": self.instance_id,
            "group_id": os.environ.get("MLX_VLM_GROUP_ID")
            or os.environ.get("MLX_VLM_WORKER_LAUNCH_ID"),
            "rank": self.group.rank(),
            "world_size": self.group.size(),
            "backend": self.backend,
            "launch_id": os.environ.get("MLX_VLM_WORKER_LAUNCH_ID"),
            "fingerprint": hashlib.sha256(signature.encode()).hexdigest(),
            "load_progress": getattr(self, "load_progress", None),
        }

    def load(
        self, model_path, adapter_path, trust_remote_code=False, progress_callback=None
    ):
        from ..load_progress import LoadingProgress

        def publish(event):
            self.load_progress = event
            if progress_callback is not None:
                progress_callback(event.copy())

        report = LoadingProgress(Path(model_path).resolve(), publish)
        report("loading_checkpoint")
        try:
            result = self._load(model_path, adapter_path, trust_remote_code, report)
        except BaseException:
            report("failed")
            raise
        report("ready")
        return result

    def _load(self, model_path, adapter_path, trust_remote_code, report):
        import mlx.core as mx

        from ..generate.diffusion import is_diffusion_model
        from ..load_progress import evaluate_model
        from ..utils import load

        path = Path(model_path).resolve()
        if (
            path.parent != self.model_dir
            or not (path / MANIFEST).is_file()
            or adapter_path
        ):
            raise ValueError(
                "Distributed workers load only coordinator-staged checkpoints without adapters"
            )
        # Retain native processor/config handling, including attach_processor.
        model, processor = load(
            str(path),
            lazy=True,
            sharding_group=self.group,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
            progress_callback=report.update,
        )
        if not callable(getattr(model, "shard", None)):
            raise ValueError("This model does not support distributed sharding")
        if is_diffusion_model(model):
            raise ValueError("Distributed serving requires an autoregressive model")
        report("sharding")
        model.shard(self.group)
        evaluate_model(model, report.update, clear_cache=True)
        model.eval()
        report("synchronizing")
        mx.eval(mx.distributed.all_sum(mx.array(1.0), group=self.group, stream=mx.cpu))
        return model, processor
