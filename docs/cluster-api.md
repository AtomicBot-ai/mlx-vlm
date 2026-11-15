# Cluster management API

Applications supply approved worker identities and endpoints to the existing
MLX-VLM server. The server owns registration validation, inference coordination,
weight preparation, lifecycle state, capabilities, and telemetry. Discovery,
macOS permissions, consent, pairing, device selection, and visualization remain
application responsibilities. This API does not browse Bonjour, install apps,
configure Remote Login, or enroll SSH keys.

Inference stays on `/v1/responses` and `/v1/chat/completions` with
`distributed: true`. The cluster API is a control plane; weight bytes use
**SSH/rsync or a dedicated JACCL RDMA transfer group**, not HTTP uploads.

## Authentication and identity

Set `MLX_VLM_SERVER_API_KEY` on the public server. Every cluster endpoint requires
its Bearer token; cluster management is disabled when no key is configured.
Each worker also runs MLX-VLM with an API key. Register its own credential in
`api_key`; this authenticates identity, health, inference, and session control.
The coordinator does not return that credential in node responses or events.
The private registry persists it in a mode-0600 file under
`MLX_VLM_CLUSTER_STATE_DIR` (default `~/.cache/mlx-vlm/cluster`). Protect this
state directory and use TLS or a trusted private transport for HTTP endpoints.

Worker API authentication is separate from SSH authentication. For remote
weights, `ssh.destination` names an **already accessible** SSH host/alias and
`ssh.python` selects that machine's Python for checksum verification. SSH uses
existing keys, known hosts, and configuration. The API never enables macOS SSH,
accepts host keys automatically, or installs credentials. The SSH account needs
write access to the worker's advertised staging directory. SSH/rsync are also
required by the optional existing CLI process launcher; neither capability is
inferred merely from successful API authentication.

Each installation persists a stable `node-id`; alternatively provision
`MLX_VLM_NODE_ID` and retain it across restarts. An address, name, rank, or Python
path is not an identity. Registration authenticates `/internal/distributed/identity`
and verifies both the approved ID and its physical-device fingerprint. Address
updates must present the same identity. Duplicate physical devices/endpoints
are rejected. The public coordinator and a rank-zero process can share one Mac;
selected memory is counted once per physical device.

## Endpoints

All schemas are in `/openapi.json`. Management operations are serialized.

| Endpoint | Behavior |
|---|---|
| `GET /v1/cluster` | State, public coordinator identity, rank-zero inference coordinator ID, selected/running members, active model, ownership, capabilities, aggregate memory |
| `GET /v1/cluster/nodes` | Registered nodes with authenticated identity, reachability, timestamped measurements and freshness |
| `POST /v1/cluster/nodes` | Verify and register an approved worker; identical registration is a no-op |
| `PATCH /v1/cluster/nodes/{node_id}` | Change endpoint, credential, name, or preconfigured SSH connection while retaining identity |
| `DELETE /v1/cluster/nodes/{node_id}` | Remove an idle registration; absent nodes are a no-op |
| `PUT /v1/cluster/members` | Select an ordered membership and backend; optional model/revision enables sharding preflight |
| `POST /v1/cluster/models/prepare` | Download/reuse and deliver verified weights to selected or explicitly named registered workers; does not launch or load a model |
| `POST /v1/cluster/start` | Attach a compatible existing collective or launch with the configured SSH launcher; optional model loads before readiness |
| `POST /v1/cluster/stop` | Drain, dispose retained sessions if explicitly requested, stop owned processes or detach external processes |
| `GET /v1/cluster/operations/{operation_id}` | Long-operation phase, progress/result, and terminal status/error |
| `POST /v1/cluster/operations/{operation_id}/cancel` | Cancel a model preparation before publication; keep verified partial chunks for retry |
| `GET /v1/cluster/events` | Throttled SSE snapshots, connection/telemetry, membership, and operation updates |

`start`, `stop`, and `models/prepare` return **202** with an `Operation`. Use
`Idempotency-Key` to retry the same operation. A different body with that key
returns `idempotency_conflict`; an overlapping management operation returns
`management_busy`. Poll the returned operation or follow events until
`status` is `succeeded` or `failed`. A 202 response alone does not imply success.
Interrupted operations become explicitly failed after a coordinator restart.

During `start`, polling the operation or subscribing to `operation.updated`
also exposes `result.progress`:

* `launching`: `stage` (`checking_endpoints`, `starting_supervisors`,
  `waiting_for_workers`, `ready`), `supervisors_ready`, `workers_ready`,
  `total_workers`, `elapsed_seconds`, and `nodes` with stable `node_id`, `rank`
  and stage. A supervisor connection is not a ready inference worker; workers
  become ready only after the existing authenticated group checks pass.
* `loading_model`: `workers_ready`, `total_workers`, `elapsed_seconds`, and
  per-node loading measurements. Worker stages are `loading_checkpoint`,
  `sharding`, `evaluating_layers`, `evaluating_remaining_parameters`,
  `synchronizing`, and `ready`. `initializing_generator` covers the remaining
  server setup after weights load. `waiting_for_load`, `unavailable` and `failed`
  are explicit rather than being reported as completed work.

For example, while a worker is materializing its parameters:

```json
{"phase":"loading_model","result":{"progress":{"elapsed_seconds":4.2,"workers_ready":0,"total_workers":2,"nodes":[{"node_id":"worker-b","rank":1,"stage":"evaluating_layers","completed_layers":12,"total_layers":47,"elapsed_seconds":3.8,"updated_at":1794657000.0}]}}}
```

The `nodes` array contains every selected worker (only one is shown above).
Layer counts advance **after** evaluation completes, preserving the existing
bounded-memory loading order. They measure that stage only, not overall load
percentage, bytes read, or an estimated remaining duration. Checkpoint parsing,
conversion, sharding and final synchronization may take time without a layer
counter. The enclosing elapsed time continues to update, while each worker's
`updated_at` identifies its last measurement. Progress is sampled at roughly
twice per second and the last snapshot is retained in the terminal operation
result; only a successful operation means the group is ready for inference.
The existing authenticated worker health/identity responses expose the same
worker measurements under `distributed_worker.load_progress`/`worker.load_progress`.

SSE records use `id`, `event`, and a JSON `ClusterEvent` containing
`id`, `type`, `timestamp`, and `data`. Reconnect with `Last-Event-ID`. Events are
bounded in memory (1024); an expired cursor, a future cursor, or a coordinator
restart produces `cluster.snapshot` so the client can reconcile current state.
`interval_seconds` (1–60, default 2) throttles telemetry refreshes per subscription.
Events and operation status contain no worker API secrets.

## Example: register, select, prepare, attach, stream

These examples use Python `httpx` against the coordinator. Discovery and user
approval happen in the application before registration. Obtain `node.id` from
an authenticated identity exchange with an already running worker endpoint;
verify the expected identity through your application's pairing mechanism.
Do not invent node IDs from addresses.

```python
import os
import time
import httpx

client = httpx.Client(
    base_url="http://127.0.0.1:8080",
    headers={"Authorization": f"Bearer {os.environ['PUBLIC_TOKEN']}"},
    timeout=30,
)

for node in approved_workers:
    client.post("/v1/cluster/nodes", json={
        "id": node.id,
        "name": node.name,
        "endpoint": node.endpoint,
        "api_key": node.worker_api_key,
        "ssh": {
            "destination": node.ssh_alias,
            "python": node.python_path,
        },
    }).raise_for_status()

model = "organization/shardable-model"
client.put("/v1/cluster/members", json={
    "node_ids": [node.id for node in approved_workers],
    "backend": "jaccl",
    "model": model,
}).raise_for_status()


def wait_operation(path, body, key):
    response = client.post(path, json=body, headers={"Idempotency-Key": key})
    response.raise_for_status()
    operation = response.json()
    while operation["status"] == "running":
        time.sleep(1)
        response = client.get(f"/v1/cluster/operations/{operation['id']}")
        response.raise_for_status()
        operation = response.json()
    if operation["status"] != "succeeded":
        raise RuntimeError(operation["error"])
    return operation

wait_operation("/v1/cluster/models/prepare", {"model": model}, "prepare-1")
wait_operation("/v1/cluster/start", {"mode": "attach", "model": model}, "start-1")

with client.stream("POST", "/v1/responses", json={
    "model": model,
    "input": "Explain why the sky is blue.",
    "distributed": True,
    "stream": True,
    "resumable": True,
    "max_output_tokens": 256,
}) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        print(line)
```

For a coordinator-local worker, `ssh` can be omitted only when its authenticated
physical identity matches the coordinator. Files are copied locally to that
worker's own staging root and verified identically. For remote workers,
registration can precede SSH configuration, but preparation/start returns
`ssh_transport_required` until the registration supplies working SSH access.
There is no HTTP checkpoint-upload endpoint or separate worker-access service.

`models/prepare` accepts optional `node_ids` (one or more registered nodes), so
weights can be delivered before selecting or starting an inference group. It
reports resolve/hash/copy/verify phases and per-node readiness. During a copy,
`operation.updated` events carry live rsync measurements in `data.result.progress`:

```json
{"node_id":"worker-b","snapshot":"sha256-digest","bytes_transferred":536870912,"bytes_reused":0,"total_bytes":2147483648}
```

The operation's `phase` is `transferring`; the same progress object is available
from `GET /v1/cluster/operations/{operation_id}`. Updates arrive during the copy,
at most twice per second per worker plus a final measurement. Model preparation
inside `start` uses the same events. Counters are cumulative for that worker's
copy attempt and reset for a new operation or worker.

`bytes_transferred` sums rsync's logical file-copy progress, including blocks
reconstructed by its delta algorithm; it is not a network-traffic counter.
`bytes_reused` counts whole files that rsync found identical with `--checksum`,
separately from copied bytes. `total_bytes` is the manifest's checkpoint size.
Copy percentage is `100 * (bytes_transferred + bytes_reused) / total_bytes`;
for a fresh destination, `bytes_reused` is zero. A previously verified snapshot
reports zero copied bytes and all bytes reused in `worker_ready`.

Copying 100% is followed by `verifying_worker`; only `worker_ready` confirms
checksum verification and atomic publication. Nonzero exits, timeouts and checksum
failures remain failed operations, even if the copy counter reached 100%.
The result includes the content digest and each worker's verified local path.
Weight delivery does not validate a sharding count; selection and inference
startup perform that separate validation. A transfer failure does not fail an
unrelated running inference group. Existing corrupt committed snapshots are
preserved for diagnosis; partial snapshots cannot be loaded.

## RDMA delivery and verified snapshot reuse

All transports use one preparation path for checkpoint validation, sealed
snapshots, partial verification and atomic publication. Only the payload delivery
step differs. `PrepareModel.transport` accepts `ssh_rsync` (the default),
`jaccl_rdma` (require RDMA), or `auto` (try a configured direct RDMA pair, otherwise
use SSH/rsync and emit `transport_fallback` with a `fallback_reason`). Automatic
fallback is permitted only for a missing RDMA pair, an unavailable JACCL backend/device
or coordinator address, or a recognized native connection failure/timeout
**before payload delivery**. Helper startup, protocol, configuration and
unexpected backend errors fail visibly instead of being relabelled as transport
unavailability. A checksum failure, lost worker, cancellation or mid-copy error fails
the operation; it never silently retries through another transport. Submit a new
operation to resume after addressing the failure.

Register the direct pair under the existing SSH access configuration:

```python
client.patch(f"/v1/cluster/nodes/{worker_id}", json={
    "ssh": {
        "destination": "approved-worker-alias",
        "python": "/worker-specific/venv/bin/python",
        "rdma": {
            "local_address": "192.168.10.1",
            "local_device": "rdma_en4",
            "remote_device": "rdma_en5",
        },
    },
}).raise_for_status()

operation = wait_operation("/v1/cluster/models/prepare", {
    "model": model,
    "node_ids": [worker_id],
    "transport": "auto",
}, "rdma-prepare-1")
```

These are deployment-supplied interfaces, not discovered or synthesized wiring.
The local address must be an IPv4 address on this coordinator, reachable by the
peer. JACCL, enabled RDMA and a compatible direct link must already be available.
The worker identity advertises `checkpoint_transfer`; backend availability does
not prove a working link. An actual isolated two-rank challenge establishes
transfer readiness. Its result is separate from inference RDMA validation and
negotiated link speed.

SSH still authenticates the helper, carries control/manifest/progress messages,
and verifies that the remote physical device matches its registered HTTP
identity. Strict host-key checking is required. Helpers are supplied as source
and use the specified Python environment's MLX, so paths can differ and no app
installation is performed. **Weight payloads use RDMA**, not the SSH connection.
RDMA payload traffic is not encrypted by this implementation; use a trusted,
dedicated link. rsync remains the encrypted alternative.

The transfer group never joins an inference collective. A pool retains at most
eight worker pairs, with a five-minute idle limit, reusing their processes and
group across files and preparations. Each pair has its own ephemeral coordinator
port. Multiple targets are served sequentially as independent two-Mac transfers;
they are not one three-rank inference group. EOF, lost control leases, cancellation
or shutdown terminate only owned transfer helpers. A failed pair is discarded;
the next attempt creates a fresh group. External inference processes stay alive.

The chunk path creates independent private snapshots, with APFS copy-on-write
clones when available. An unsupported clone operation or a cross-filesystem
source uses a streamed copy; permission, disk-space and other filesystem errors
fail preparation immediately. It computes whole-file SHA-256 and ordered 16 MiB chunk
digests during the same initial read, then persists a verified manifest. Files
and directories are made read-only. A receipt records device/inode identity,
size, nanosecond modification/change times and modes for the entire snapshot.
Reuse requires that receipt and the sealed tree to match. A changed tree is
reverified; corruption fails rather than overwriting the published snapshot.
Original source files are never chmod'ed or hard-linked into this cache.

This cache assumes the server account and its private state are trusted. It is
not protection against an attacker controlling that same OS account. Checking
only a filename, size, or mtime is insufficient. Receipts are only produced after
verification and are never accepted from an unverified source directory.
Receipt reuse does not detect silent storage corruption that leaves filesystem
metadata unchanged; checking for that requires a separate full readback audit.

On delivery, the receiver hashes every incoming chunk against the authenticated
manifest, overlaps hashing and disk writes using bounded queues, flushes file
data (`F_FULLFSYNC` on macOS), and atomically publishes only after both helpers
succeed. Resume rechecks bytes in partial files instead of trusting a stale
completion bitmap. A valid destination snapshot reports `transport:
"verified_cache"`, zero transferred bytes, and the entire checkpoint reused.
The source manifest and destination receipts survive coordinator restarts;
helper processes do not. Preparation and subsequent inference staging share
the same snapshot store and verification receipts, including when switching
between RDMA and SSH/rsync. Snapshots from the former whole-file staging format
remain intact and loadable; a new preparation creates the chunk manifest once.

Progress uses the existing operation/events endpoints:

| Field | Meaning |
|---|---|
| `transport`, `node_id`, `snapshot` | Actual transport, registered target, immutable snapshot identity |
| `bytes_read` | Source file bytes read during this transfer attempt |
| `bytes_received` | Payload bytes evaluated at the receiving RDMA rank |
| `bytes_verified` | Verified received chunks plus verified reused chunks |
| `bytes_reused` | Valid bytes already present at the destination |
| `bytes_transferred`, `total_bytes` | Delivered logical payload and manifest total |
| `bytes_checked` | Bytes inspected while validating a partial file, including invalid chunks |
| `elapsed_seconds`, `throughput_bytes_per_second` | Measured attempt time and logical transfer progress/time |
| `rate_kind` | `logical_payload` for RDMA, `logical_copy_progress` for rsync; neither is a wire-traffic counter |

RDMA updates are throttled to roughly five per second plus phase boundaries;
rsync retains its twice-per-second interval. `verifying_source`,
`checking_partial`, `connecting`, `connected`, `transferring`, `publishing`, and
`worker_ready` distinguish hashing, connection setup, delivery and durable
publication. Counters reset for each attempt/target; source preparation has its
own counters. Copy completion alone is not readiness.

When the coordinator's stderr is a terminal, preparation also displays `tqdm`
byte-progress bars for the source and each worker. This covers both explicit
`models/prepare` calls and preparation triggered by inference. Bars show the
current phase and actual transport; transfer speed and ETA use the reported
logical-copy/payload rate, not measured cable traffic. Unknown totals display byte
counts without a percentage. Verification and publication remain separate phases,
and only `worker ready` indicates readiness. Redirected logs do not contain bars;
applications continue to consume the operation/events endpoints above.

Cancellation returns 202 and is observed through the original operation. A
cancelled preparation finishes as `failed` with `preparation_cancelled`; a retry
needs a new idempotency key. Publication is a commit point: cancellation after
that point is rejected with `cancellation_too_late`. Already completed operations
are returned unchanged. Explicit SSH/rsync, RDMA and automatic preparations all
support cancellation, including while waiting for the staging lock. This endpoint
does not cancel inference or other lifecycle operations (`cancellation_unavailable`).

The first snapshot still pays the initial hash/copy cost. Manifest reuse and warm
helpers remove repeated work, not that initial verification. Published benchmark
results must distinguish first preparation, manifest reuse with fresh helpers,
warm-helper transfer, and a destination cache hit. Optional independent disk
readback is an audit; inline chunk verification does not reread storage after
every write.

## Lifecycle and ownership

Registered nodes, selected members, and the running collective are distinct.
An identical member selection is idempotent. Changing running membership or a
running node's connection returns `group_busy`; first stop/drain the group,
then update the selection and start a new group. Never add or remove ranks from
an active collective. Model dimensions and architecture-specific constraints
are checked for the selected count: three devices are not automatically valid
for a model whose attention dimensions divide only by two/four/five.

`start` with `mode: "attach"` requires externally launched workers to present the
same nonempty `MLX_VLM_GROUP_ID`, backend, world size, and rank ordering. Code and
runtime fingerprints must agree. If they need reconfiguration, the failed
operation supplies `external_reconfiguration_required` and the required rank
plan. Applications/process supervisors launch each worker using its own Python
and filesystem paths; MLX-VLM does not install or manage that external process.

`mode: "launch"` reuses the existing `cluster.json` SSH launcher/hostfile. The
selected endpoints must match that launcher configuration; changing its physical
hostfile/topology requires reconfiguration while stopped. It does not synthesize
RDMA wiring from IP addresses. Topology is ultimately validated by initializing
the configured MLX backend and successfully running collectives, not discovery.

Managed launches honor the experimental, default-off `launch.fast_sync` boolean
in `cluster.json`. To override it without restarting the public server, first
stop the group and wait for that operation to succeed, then start fresh workers:

```json
{"mode": "launch", "fast_sync": true, "model": "organization/shardable-model"}
```

`fast_sync` accepts only a boolean or `null`; omitting it (or using `null`) inherits
the configured default. The override applies only to this launch and does not
change the default for future starts. Use `false` to disable it. A start request
while the group is running still fails with `already_running`; `mode: "attach"`
rejects a non-null `fast_sync` because external workers are already initialized.
`GET /v1/cluster` exposes the active managed group's effective `fast_sync`, or
`null` when stopped or attached externally. See [distributed launch configuration](distributed-serving.md)
for MLX's deadlock warning. All workers receive the setting before Python starts.

`stop` first rejects new distributed admissions, then waits for active requests
to drain, up to `drain_timeout_seconds` (default 60). A timeout is observable and
never silently kills active inference. Retained continuations return
`retained_continuations` unless `discard_continuations: true` was supplied.
A repeated `start` returns a failed operation with `already_running` without
changing the existing group. A rejected or cancelled `stop` before checkpoint
disposal or process shutdown restores the prior readiness/admission state; its
operation can fail while the cluster remains healthy. Once disposal or shutdown
begins, a failure leaves admissions blocked because ranks may have changed.
Explicit disposal synchronizes native checkpoint/cache disposal across ranks.
A failed collective cannot acknowledge cache disposal. Explicit discard then
invalidates coordinator handles, preserves the failure marker, and reports
`collective_disposal_acknowledged: false`; external owners also receive
`external_restart_required: true`. Fresh rank processes are required to recover.
Only processes launched by this server are stopped; externally managed workers
are detached and remain running. Local inference remains independent.

A lost rank/semantic divergence marks the group failed. Replace the entire
affected collective, then start/attach again. Recovery checks fresh process
instance IDs before clearing the failure marker; a healthy HTTP endpoint alone
does not prove a broken collective recovered. Restarting the coordinator keeps
registrations/selection but requires explicit reattachment to external workers.

## Continuations, progress, and capabilities

Distributed Responses use the existing `resumable`/`continuation` contract. Save
the latest event's continuation object, close the stream, wait for the pause to
finish, and resend the original request with that object in `continuation`.
Tokens already generated are replayed from the native decoder history; retained
KV/recurrent caches resume decoding. Prefill, pause/resume, expiry, disposal,
and cache-budget decisions are synchronized at collective scheduling boundaries.
Concurrent replacement readers return retryable `continuation_busy` rather than
claiming ranks in different orders. Continuations are process-local, tenant and
model scoped; they do not survive group/coordinator restarts.

`capabilities.resumable` is negotiated against the active workers' session
protocol. Streaming/batching and progress support are advertised separately.
Stored Responses retrieval/deletion and `previous_response_id` are **not**
supported for distributed inference; their capability fields remain false.
Reasoning, tool events, compaction capsules, usage, and terminal events retain
the normal Responses contract. MiniMax thinking delimiters are parsed across
arbitrary token boundaries, including closing-only continuations.

Local serving and distributed workers both use `ResponseGenerator`'s existing
prefill measurements. The Responses endpoint exposes them as events:

```json
{"type":"response.prompt_progress","response_id":"resp_…","status":"in_progress","processed_tokens":512,"total_tokens":2048}
{"type":"response.generation_progress","response_id":"resp_…","output_tokens":16,"timings":{"predicted_per_second":32.0}}
```

Prefill uses `started`, `in_progress`, and `completed` with monotonic, measured
counts, emitted while actual prefill runs and before the first generated token.
The direct `stream_generate` fallback has no scheduler prefill measurements and
does not emit prompt-progress events.
Generation timing fields appear only when measured. `timings.predicted_per_second`
uses the same field name as regular text events and native generation metrics.
The coordinator forwards rank zero's measurements. Timing or progress frequency
differences do not count as semantic divergence; token,
reasoning/tool, terminal, and collective decisions still do.

## Telemetry and errors

Per-node system memory reports total/used **bytes** separately from MLX
active/cache/peak allocation. Measurements include timestamps, OS, MLX and
MLX-VLM versions, backend availability, health, and freshness. Missing values
remain null; selected aggregates expose `complete`, `missing_node_ids`, and
`stale_node_ids`. Measurements older than 15 seconds are marked stale. Only
selected physical devices count toward the aggregate, including rank zero once.

Link properties come from `ibv_devinfo` when available. Unknown negotiated rates
remain null. Link-active state is not RDMA proof: `rdma_validated` becomes true
only after this JACCL worker has successfully verified inference collectives.
`coordinator` identifies the public server; `inference_coordinator_id` identifies
the registered rank-zero worker for a client-side “Main” badge.

Management conflicts use `{"detail": {"code", "message", "retryable", "action"}}`.
Failed long operations place the same `ClusterError` under `operation.error`.
Authentication/validation errors also retain standard FastAPI status semantics.

| Code | Client action |
|---|---|
| `identity_mismatch`, `duplicate_device` | Recheck approved identity; do not create a duplicate from a new address |
| `management_busy`, `group_busy` | Wait for the operation, or drain/stop before changing membership |
| `unsupported_member_count`, `backend_unavailable` | Choose a supported model/count/backend |
| `ssh_transport_required` | Supply preconfigured SSH access independently of API credentials |
| `external_reconfiguration_required`, `launch_reconfiguration_required` | Reconfigure the external group or existing hostfile using the returned plan |
| `drain_timeout`, `retained_continuations` | Finish active streams or explicitly discard retained sessions |
| `restart_required`, `transition_failed` | Inspect the operation; replace failed ranks before retrying |
| `coordinator_restarted` | Reconcile workers and retry the interrupted operation with a new key |

## Application migration

| Application responsibility before | Supported server responsibility now |
|---|---|
| Rank request fan-out and stream relaying | Existing inference endpoints with `distributed: true` |
| Coordinated batch admission and cancellation | Shared distributed scheduler and native sessions |
| Parsing worker logs for status/memory | Cluster/nodes/operations endpoints and throttled events |
| Managing rank membership implicitly | Stable registrations, explicit selection and safe group transitions |
| Downloading/synchronizing weights itself | `models/prepare`, verified snapshots, SSH/rsync or isolated RDMA delivery |
| Resume implementation or regeneration | Existing Responses continuation contract across ranks |

Applications still own discovery, pairing/consent, device UI, and their own
external process supervision. SSH trust/access and backend/network preparation
remain deployment responsibilities. No Nativ fields or native onboarding code
are part of MLX-VLM.
