# Copyright © 2026 Neywa Labs Inc.

"""Rank-zero admission order for the native continuous batch scheduler.

Only the generation thread uses the collective group. HTTP threads may arrive
in any order; a row is admitted only when every rank has prepared that row.
"""

import hashlib
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from .cluster_types import SpeculativeDraft


@dataclass
class RequestScope:
    request_id: str
    sequence: int = 0
    draft: SpeculativeDraft | None = None

    def next_id(self):
        self.sequence += 1
        return f"{self.request_id}:{self.sequence}"


request_scope = ContextVar("mlx_distributed_request", default=None)


def gather_json(group, value):
    """Exchange CPU control messages between completed model steps."""
    import mlx.core as mx

    # Native decoding prefetches the next token asynchronously. Finish that
    # work before issuing control collectives on the same group from a CPU
    # stream, so different rank timings cannot reorder collective operations.
    mx.synchronize()
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    lengths = mx.distributed.all_gather(
        mx.array([len(data)], dtype=mx.int32), group=group, stream=mx.cpu
    ).tolist()
    width = max(lengths)
    if width > 4 * 1024 * 1024:
        raise RuntimeError("Distributed scheduler control message exceeds 4 MiB")
    packed = mx.array(list(data) + [0] * (width - len(data)), dtype=mx.uint8)
    rows = mx.distributed.all_gather(packed, group=group, stream=mx.cpu)
    rows = np.asarray(rows).reshape(group.size(), width)
    return [json.loads(row[:length].tobytes()) for row, length in zip(rows, lengths)]


def request_signature(request, batch_key):
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                digest.update(str(key).encode())
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            array = np.asarray(value)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())

    visit(request.raw_inputs)
    from .sessions import InferenceSession

    visit(
        [
            batch_key,
            request.prompt_tokens,
            request.args.max_tokens,
            getattr(request.args, "emit_prompt_progress", False),
            isinstance(request.rqueue, InferenceSession),
        ]
    )
    return digest.hexdigest()


def schedule(states, capacity):
    """Use rank 0's order, readiness intersection, and cancellation union."""
    if any(state.get("phase") != "schedule" for state in states):
        raise RuntimeError("Distributed generation phases diverged")
    if any(state["error"] for state in states):
        raise RuntimeError(
            "A distributed scheduler failed: "
            + str(next(state["error"] for state in states if state["error"]))
        )
    if any(state["active"] != states[0]["active"] for state in states[1:]):
        raise RuntimeError("Distributed active batches diverged")
    cancelled = set().union(*(state["cancelled"] for state in states))
    cancelled_requests = set().union(*(state["cancelled_requests"] for state in states))
    ready = [dict(state["ready"]) for state in states]
    selected = []
    for request_id, signature in states[0]["ready"]:
        if request_id.split(":", 1)[0] in cancelled_requests:
            continue
        if not all(request_id in rank for rank in ready):
            continue
        if any(rank[request_id] != signature for rank in ready):
            raise RuntimeError(
                "Distributed preprocessed inputs or sampling settings differ"
            )
        if capacity is None or len(selected) < capacity:
            selected.append(request_id)
    return (
        selected,
        cancelled,
        cancelled_requests,
        any(state["stop"] for state in states),
    )


class DistributedBatchScheduler:
    def __init__(self, worker, exchange=None):
        self.worker = worker
        self.exchange = exchange or (lambda value: gather_json(worker.group, value))
        self.pending = {}
        self.signatures = {}
        self.error = None

    def fail(self, error):
        self.error = str(error)

    def next(self, generator, groups, limit):
        items, _ = generator._collect_pending_requests(active=True)
        for item in items:
            if not item.request_id or ":" not in item.request_id:
                self.fail("Missing coordinated generation request ID")
            self.pending[item.request_id] = item
            try:
                self.signatures[item.request_id] = request_signature(
                    item, generator._batch_key(item)
                )
            except Exception as error:
                self.fail(error)
        active = [
            [list(key), [[uid, info["request_id"]] for uid, info in rows.items()]]
            for key, (_, rows) in groups.items()
            if rows
        ]
        running = sum(len(rows) for _, rows in groups.values())
        capacity = None if limit is None else max(0, limit - running)
        states = self.exchange(
            {
                "phase": "schedule",
                "ready": list(self.signatures.items()),
                "active": active,
                "cancelled": sorted(generator._drain_cancellations()),
                "cancelled_requests": self.worker.cancelled_requests(),
                "stop": generator._stop,
                "error": self.error,
            }
        )
        chosen, cancelled, request_cancels, stop = schedule(states, capacity)
        self.worker.remember_cancellations(request_cancels)
        for request_id in list(self.pending):
            if request_id.split(":", 1)[0] in request_cancels:
                item = self.pending.pop(request_id)
                self.signatures.pop(request_id)
                item.rqueue.put(RuntimeError("Distributed request cancelled"))
        for _, rows in groups.values():
            cancelled.update(
                uid
                for uid, info in rows.items()
                if info["request_id"].split(":", 1)[0] in request_cancels
            )
        requests = [self.pending.pop(request_id) for request_id in chosen]
        for request_id in chosen:
            self.signatures.pop(request_id)
        if not running and not requests:
            time.sleep(0.005)
        return requests, cancelled, stop and not running and not self.pending

    def continuations(self, generator, groups, suspended, capacity):
        """Exchange intent, then use the existing native cache suspend/resume path.

        Any rank can request pause/expiry/cancellation. Resume needs agreement
        from every rank on the same reader lease. Wall clocks and cache budgets
        never independently decide a collective's membership.
        """
        import os

        from .sessions import InferenceSession

        sessions = {}
        values = {}
        for _, (_, active) in groups.items():
            for uid, info in active.items():
                session = info["rqueue"]
                if isinstance(session, InferenceSession):
                    sessions[uid] = session
                    values[uid] = (False, 0)
        for session, (_, checkpoint, uid, info) in suspended.items():
            sessions[uid] = session
            values[uid] = (True, generator._checkpoint_bytes(checkpoint, info))
        rows = []
        for uid, session in sessions.items():
            error = None
            try:
                session.check()
            except Exception as exc:
                error = str(exc)
            rows.append(
                [
                    uid,
                    session.paused,
                    session.lease,
                    error,
                    values[uid][0],
                    values[uid][1],
                    session.touched,
                ]
            )
        states = self.exchange(
            {
                "phase": "sessions",
                "rows": rows,
                "budget": max(
                    0, int(os.environ.get("MLX_VLM_CONTINUATION_CACHE_BYTES", 1024**3))
                ),
            }
        )
        if any(state.get("phase") != "sessions" for state in states):
            raise RuntimeError("Distributed session phases diverged")
        mappings = [{row[0]: row for row in state["rows"]} for state in states]
        if any(set(row) != set(mappings[0]) for row in mappings):
            raise RuntimeError("Distributed retained sessions diverged")
        decisions = {}
        resident = sum(max(mapping[uid][5] for mapping in mappings) for uid in sessions)
        budget = min(state["budget"] for state in states)
        for uid in sorted(sessions, key=lambda value: (mappings[0][value][6], value)):
            peers = [mapping[uid] for mapping in mappings]
            error = next((row[3] for row in peers if row[3]), None)
            if any(row[4] != peers[0][4] for row in peers):
                raise RuntimeError("Distributed cache suspension diverged")
            if peers[0][4] and resident > budget:
                error = "The inference checkpoint was evicted to release cache memory."
                resident -= max(row[5] for row in peers)
            paused = any(row[1] for row in peers) or len({row[2] for row in peers}) != 1
            decisions[sessions[uid]] = (paused, error)
        generator._schedule_continuations(
            groups, suspended, capacity, decisions=decisions
        )
        result = [[uid, session.distributed_state] for uid, session in sessions.items()]
        confirmed = self.exchange({"phase": "session_results", "rows": result})
        if any(state != confirmed[0] for state in confirmed):
            raise RuntimeError("Distributed session transitions diverged")
        for session in sessions.values():
            session.distributed_ack = session.distributed_state

    def verify_step(self, responses):
        values = [
            [
                response.uid,
                None if response.token is None else int(response.token),
                response.finish_reason,
            ]
            for response in responses
        ]
        states = self.exchange({"phase": "tokens", "values": values})
        if any(
            state.get("phase") != "tokens" or state != states[0] for state in states
        ):
            raise RuntimeError("Distributed token or completion decisions diverged")
        if values:
            self.worker.inference_validated_at = time.time()

    def abort(self, error, generator, groups):
        generator._load_error = error
        queued, _ = generator._collect_pending_requests(active=True)
        queues = [item.rqueue for item in [*self.pending.values(), *queued]]
        queues += [
            info["rqueue"] for _, rows in groups.values() for info in rows.values()
        ]
        for queue in queues:
            queue.put(error)
            queue.put(None)
        self.pending.clear()
        self.signatures.clear()
