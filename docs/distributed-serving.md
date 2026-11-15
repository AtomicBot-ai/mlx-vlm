# Per-request distributed serving

One public MLX-VLM server can handle both local and distributed models through
the existing API routes. Omit `distributed`, or set it to `false`, to use the
normal single-device loader. Set it to `true` to use a configured worker group.
The authenticated [cluster API](cluster-api.md) manages registrations, selected
members, lifecycle, telemetry, and SSH/rsync weight preparation without a restart.
The coordinator remains outside that collective group, with a separate worker
process on each participating Mac, including the main Mac.

This is opt-in: an unconfigured distributed request returns an error instead
of silently loading the model locally. Models need an implemented `Model.shard`
method; a large checkpoint is not automatically a shardable checkpoint.

## How a request runs

For a three-Mac deployment, the public server and rank 0 worker are separate
processes on the main Mac. Ranks 1 and 2 run on the other Macs:

```text
Client -> public server (coordinator)
            |
            +-- distributed omitted/false -> normal local generation
            |
            +-- distributed true -> same request to all workers
                                      rank 0 <-> rank 1 <-> rank 2
                                       |   MLX collective operations
                                       v
                          rank 0 tokens -> coordinator -> client
                          peer streams -> coordinator consistency checks
```

1. **Route and reserve the model.** The coordinator reads `distributed` on the
   existing endpoint. Distributed requests for the same model share a model
   lease, so they can run together. A request for a different model waits for
   active requests and their streams to drain before switching the group.
2. **Prepare the checkpoint once.** The coordinator reuses cached files or
   downloads missing files on the main Mac, stages a snapshot, and transfers it
   to peers that need it. Checksums must pass before workers load it. Each Mac
   stores a full snapshot on disk; the native `Model.shard` implementation
   determines how model parameters and computation are distributed in memory.
3. **Admit identical batches.** All workers receive the same request ID and
   sampling seed. Their generation threads exchange ready IDs, input hashes,
   active rows, and cancellations at scheduling boundaries. Every rank follows
   rank 0's ready order and admits only requests prepared identically everywhere.
   This keeps collective operations in the same order despite HTTP timing
   differences. The coordinator routes requests; it does not participate in
   distributed model steps.
4. **Generate and stream continuously.** The existing native `BatchGenerator`
   performs prefill and decoding. New compatible requests join as slots become
   available, and completed rows leave without waiting for the whole batch.
   Workers verify token and completion decisions before publishing them. The
   coordinator forwards rank 0's SSE events while consuming and comparing peer
   streams; it releases terminal success only after all ranks agree.
5. **Clean up together.** An ordinary client disconnect cancels its request
   across all ranks. A resumable Responses stream instead pauses the existing
   native decoder/cache session across ranks, retaining it for continuation. The model lease remains held until cleanup finishes, so a model switch
   cannot overlap old work. A rank failure or disagreement blocks further
   distributed requests until the group is recovered; local serving remains
   available.

For example, if two requests share a model and one finishes early, its slot can
accept a third request while the longer request keeps generating. All ranks
make that admission together. `--max-num-seqs` controls worker admission and
accepts positive integers; the native batch limits and tested capacities are
listed under [Current boundaries and recovery](#current-boundaries-and-recovery).

## Requests

Send this body to `/v1/chat/completions` on the public server, using its usual
Bearer authentication. Replace the example model ID with a compatible model:

```json
{
  "model": "organization/shardable-model",
  "distributed": true,
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 128,
  "stream": true
}
```

The same flag works on `/chat/completions`, `/v1/responses`,
`/v1/responses/input_tokens`, and `/v1/responses/compact`. With an OpenAI-compatible
SDK, put `distributed` in `extra_body`. Optional `distributed_revision` selects a
Hub branch, tag, or commit. It applies only to distributed Hub model IDs.

Each request can choose a different model. Workers load the newly selected
checkpoint through the normal server model cache, then shard it across their
group. A subsequent request with `distributed: false` still uses the public
server's normal local cache. Multiple distributed models are not kept resident
simultaneously; the group switches its active model as needed.

## Cluster setup

Use the same MLX-VLM checkout and dependency versions on every worker. Configure
MLX's hostfile/backend for the machines and test that network separately. With
a `launch` section in `cluster.json`, starting the public server also starts
its workers through `mlx.launch` over SSH. No separate worker command or
`cluster apply` step is needed. The server does not enroll Macs, configure RDMA,
or install packages; those remain one-time setup steps.

Requirements:

- A dedicated collective group, with one worker endpoint per rank.
- An absolute staging directory on each machine, writable by its worker and
  transfer user. Each machine needs room for the full checkpoint. A worker can
  specify its own `model_dir` and `python`; the legacy CLI defaults remain shared.
- Noninteractive SSH and `rsync` from the coordinator to remote workers for
  weights. Automatic process launch also uses SSH. MLX-VLM never configures
  macOS Remote Login, enrolls SSH keys, or changes SSH trust.
- Authenticated worker endpoints, independent of SSH access. API registrations
  can supply individual worker keys; file configurations also support a shared
  key through `api_key_env`, distinct from the public server's key if desired.
  Keep worker endpoints private and send inference through the coordinator.
- Identical worker settings. Health checks compare source, MLX version, runtime
  settings, rank count, and backend before loading a model.

For example, save this as `cluster.json` on the main Mac. Host names and paths
are examples to replace with your deployment's values:

```json
{
  "workers": [
    {"url": "http://127.0.0.1:8283"},
    {"url": "http://mac-two.internal:8283", "ssh": "mac-two"},
    {"url": "http://mac-three.internal:8283", "ssh": "mac-three"}
  ],
  "model_dir": "/Users/Shared/mlx-vlm-distributed-models",
  "shared_lock": "/Users/Shared/mlx-vlm-distributed.lock",
  "python": "/Users/Shared/mlx-vlm/.venv/bin/python",
  "api_key_env": "MLX_VLM_DISTRIBUTED_API_KEY",
  "timeout": 7200,
  "launch": {
    "hostfile": "hosts.json",
    "backend": "jaccl",
    "cwd": "/Users/Shared/mlx-vlm",
    "host": "0.0.0.0",
    "port": 8283,
    "max_num_seqs": 8,
    "fast_sync": false,
    "startup_timeout": 120,
    "shutdown_timeout": 10
  }
}
```

Omit `ssh` only for a worker sharing the coordinator's filesystem. Optional
per-worker `model_dir` and `python` override staging defaults; `api_key` overrides
the shared environment credential (keep configuration files private). Worker URLs
are HTTP(S) origins reachable from the coordinator; they can also be local SSH
tunnel endpoints. Optional `ssh_options` is an array of arguments, for example
`["-F", "/path/to/ssh-config"]`. Use a staging path without spaces or shell
metacharacters. Give each independent worker group its own staging root and
lock file. All coordinators for the *same* group must share that filesystem lock;
an API client should not acquire the coordinator's lock itself.

The launch hostfile is resolved relative to `cluster.json`. Its SSH entries must
match `workers` in order, with one worker per host; use `127.0.0.1` for the local
entry. Keep the existing MLX IP/RDMA topology in that file. `python` and
`launch.cwd` must be absolute paths available on every Mac. The example assumes
you have already run `uv sync` in the checkout on each machine. Automatic launch
supports `jaccl` and `ring`.

For independent groups on the same hosts, set `launch.starting_port` to a free
collective control port (default `32323`), and use distinct worker HTTP ports.
This forwards MLX's `--starting-port` option; it does not change `launch.port`.

`launch.fast_sync` is an **experimental**, default-off boolean. Set it to `true`
to enable MLX's faster CPU/GPU synchronization for every managed worker. The
launcher sends `MLX_METAL_FAST_SYNCH=1` (or `0` when disabled) through the supervisor
before the worker Python process starts and imports MLX. This overrides inherited
host/hostfile settings; configure it with `launch.fast_sync`, not `launch.env`.
It applies both at public-server startup and to cluster API `mode: "launch"`.
Externally managed workers attached with `mode: "attach"` are unaffected.

MLX [caches this setting on first use](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/utils.h#L203-L206),
so changing an environment variable during generation is insufficient. To change
it without restarting the public server, drain/stop the group through
`POST /v1/cluster/stop`, wait for success, then call `POST /v1/cluster/start` with
`{"mode": "launch", "fast_sync": true}` (or `false`). Include `model` to load a
checkpoint during startup. The override applies only to that launch; omitting it
uses the configured `launch.fast_sync` default. `GET /v1/cluster` reports the
active managed group's `fast_sync` value, or `null` for stopped/external groups.
Changing the default in `cluster.json` still requires restarting the public server.
MLX [warns that fast synchronization can
deadlock and leave the GPU wedged](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/distributed.rst).
Keep it off unless testing a known workload; a short successful run does not
establish long-running stability. This option exposes the existing MLX setting;
it does not add another speedup to clients already enabling it.

Start only the public server, using your existing keys:

```bash
cd /Users/Shared/mlx-vlm

export MLX_VLM_DISTRIBUTED_API_KEY="$WORKER_TOKEN"
export MLX_VLM_SERVER_API_KEY="$PUBLIC_TOKEN"

uv run --no-sync python -m mlx_vlm.server \
  --distributed-config cluster.json \
  --host 127.0.0.1 \
  --port 8080
```

Before accepting requests, the server acquires a separate launcher ownership
lock, checks for existing services at the worker endpoints, and launches the
group. It waits for authenticated health responses with matching code/settings,
the configured backend and rank count, and a unique identity for this launch.
Startup fails if that process exits or the readiness deadline expires. Existing
or partly running groups are never adopted or replaced: stop your previous
group yourself, or omit `launch` to use externally managed workers.

Worker credentials travel through the launcher's stdin after every supervisor
has acknowledged startup, rather than appearing in command arguments. A scoped
SSH wrapper applies `ssh_options` and requires noninteractive authentication
with a previously trusted host key. Missing SSH access fails startup; it does
not prompt for passwords or change SSH trust. Worker logs appear in the public
server's log with an `mlx.launch` prefix.

Workers do not inherit the public server's preload, draft, API-key or serving
tuning settings. To set shared worker tuning, use an optional `launch.env`
object, for example `{"PREFILL_STEP_SIZE": "1024", "KV_BITS": "4"}`. The worker
key and launch identity are managed automatically. Keep this object small:
the complete stdin setup message has a 4000-byte limit. Supervisors disable PTY
echo and canonical input buffering before acknowledging readiness.

On normal shutdown, Uvicorn finishes active requests before the server stops
its owned worker group. Each worker has a separate stdlib supervisor, so a
native collective cannot block its stop handler. A worker that does not exit
within `shutdown_timeout` is killed. Supervisors also stop workers after 30
seconds without a coordinator heartbeat, covering abrupt coordinator or
launcher loss. Startup failure and cancellation use the same cleanup path.

If the launcher exits during serving, distributed requests are marked failed;
local serving stays available. There is no automatic restart during inference.
Change membership by editing the configuration and restarting the public
server after draining requests. The launcher lock is held for the owned group's
lifetime and its file is not deleted on exit.

### Externally managed workers

Omit `launch` from `cluster.json` to retain the separate worker-start workflow.
The public server then neither starts nor stops worker processes, and the
existing request-time health checks remain in place.

Configure `MLX_VLM_SERVER_API_KEY` with the same worker token in each worker's
launch environment. No preload, adapter, or draft-model environment settings
should be inherited by workers. Then launch the worker group using your existing
MLX hostfile and environment provisioning:

```bash
mlx.launch --hostfile /path/to/hosts.json --backend jaccl \
  --python /Users/Shared/mlx-vlm-venv/bin/python \
  --cwd /path/to/mlx-vlm -- \
  -m mlx_vlm.server --distributed-worker \
  --distributed-backend jaccl --max-num-seqs 4 \
  --distributed-model-dir /Users/Shared/mlx-vlm-distributed-models \
  --host 0.0.0.0 --port 8283
```

The launcher must propagate the worker environment to **every rank**; exporting
a variable only in the coordinator's terminal is not sufficient for SSH workers.
Use your existing protected environment file or remote launch wrapper for keys.
Use `ring` instead of `jaccl` in both flags for an appropriately configured ring
hostfile. There must be at least two ranks.

In the public server's environment, set `MLX_VLM_SERVER_API_KEY` to its client
token and `MLX_VLM_DISTRIBUTED_API_KEY` to the worker token. Start the public server:

```bash
python -m mlx_vlm.server --distributed-config /path/to/cluster.json \
  --host 127.0.0.1 --port 8080
```

This server can still use ordinary local options such as `--model` to preload a
local model. Account for the combined GPU memory used by that model and the
main Mac's separate distributed worker. The coordinator does not automatically
evict its local model to make room for the worker.

## Download, transfer, and loading

On the first distributed request for a model/revision:

1. The coordinator checks that the workers form one compatible group.
2. It resolves a local model directory or uses MLX-VLM's normal Hub download
   path. Existing cached files are reused; missing files are downloaded on the
   main Mac. Hub credentials stay there.
3. It validates the architecture's sharding support and weight index, hashes
   up to eight files concurrently, and materializes a snapshot with a content-derived
   name. For Hugging Face cache blobs, the published LFS SHA-256 recorded in the
   [blob's name](https://huggingface.co/docs/huggingface_hub/package_reference/file_download#hf_hub_download)
   is the expected checksum. The coordinator still checks the bytes:
   a filename alone cannot establish cache integrity. Git SHA-1 blob IDs are not
   treated as SHA-256. Hub symlinks become private copies on each machine.
4. It reuses a verified snapshot on each peer or transfers it with the selected
   JACCL RDMA or SSH/rsync transport. For the first RDMA-capable peer, up to
   eight files stream in parallel: each source buffer is read once, hashed, and
   sent through the dedicated transfer group. Two bounded CPU buffers per file
   keep hashing ahead of transmission. The receiver verifies chunk hashes while
   writing, then compares its completed chunk manifest with the sender's manifest
   over the authenticated SSH control channel. No checkpoint is published until
   every file passes its whole-file and chunk checks;
   later peers use the completed manifest. Bounded RDMA receiver buffers overlap
   verification and sequential disk writes; macOS uses uncached streaming writes
   to avoid retaining a checkpoint's worth of file-cache pages. Checksums and
   durable writes must complete before atomic publication; cancellation or
   corruption never makes a partial snapshot loadable.
5. It sends the same request, tenant, and sampling seed concurrently to all
   ranks, translating the immutable snapshot to each worker's local path. Workers use the shared native model/processor loader lazily,
   call `Model.shard`, materialize parameters, and synchronize before generation.

Workers do not resolve Hub IDs themselves. Later requests reuse the prepared
snapshot. The coordinator resolves each model/revision once per process; pin a
commit for reproducibility and restart the coordinator when intentionally
refreshing a moving Hub branch or an edited local directory. Staged snapshots
are immutable and are not automatically deleted. Unchanged, sealed snapshots reuse
verification receipts without rereading weights. A fresh snapshot still requires
whole-file and chunk verification; Hugging Face does not supply chunk checksums.
For the first RDMA peer, verification uses the same source buffers as transmission,
removing the separate preparatory read. APFS copy-on-write clones isolate the
source without copying its contents; other filesystems require a bounded private
copy first. SSH/rsync still waits for the complete source manifest. During RDMA
streaming, `bytes_read`, `bytes_received`, `bytes_verified` and `bytes_reused`
report actual work under the transfer phase. Interrupted streams inspect actual
partial bytes and compare them with newly calculated source chunk hashes before
reusing them; changed chunks are retransmitted.
Workers currently receive the complete checkpoint. Delivering only rank-local
weights requires a model-aware tensor partition format: Hub shard filenames do
not identify inference ranks.

## Current boundaries and recovery

Distributed requests support streaming and continuous batching. Set `stream:
true` for SSE on Chat Completions or Responses; ordinary JSON responses use the
same batch scheduler. Set the same positive `--max-num-seqs` on every worker
(the distributed default is 4). Values such as 3, 12, or 16 are also accepted;
4 and 8 are the tested capacities. The native decoder retains its default
limit of 32 decoding rows per compatible group and prefills up to 8 rows at a
time. A larger `--max-num-seqs` admits more requests but does not raise those
internal limits. Memory use also depends on the model and context lengths.
Compatible requests for the same model
can join an active batch as slots become available. Different sampling settings
use the native scheduler's separate groups. A model switch waits for current
requests, including streams, to drain; later requests cannot starve the switch.

The native generation thread gathers prepared-request identities and input
hashes at each scheduling boundary. All ranks use rank 0's admission order,
admit only requests prepared identically everywhere, combine cancellations,
and verify token/completion decisions before emitting tokens. Late arrivals do
not make workers schedule different collectives. Default requests share a
sampler seed, with independent row keys, so random per-request seeds do not
accidentally prevent batching.

The coordinator forwards rank 0's SSE events as they arrive and drains every
other rank concurrently. It compares normalized stream contents and withholds
the terminal success event until all ranks finish consistently. An ordinary
disconnect cancels that request across ranks and retains the model lease through
cleanup; a resumable disconnect synchronizes native checkpoint suspension.
Other active rows continue. Measured progress events come from rank zero and
are excluded from semantic divergence checks. Stream failures produce an SSE error and mark the
group for recovery.

Responses requests remain stateless (`store: false`); provide full history or a
compaction capsule instead of `previous_response_id`. Streaming Responses can
use `resumable: true` and the existing `continuation` object; resume retains the
native cache and replays decoder history without generating old text again.
Close the current reader before resuming (concurrent claims return
`continuation_busy`). Paused/completed retained sessions pin the model until
expiry or explicit disposal with cluster stop. Stored retrieval/deletion,
`previous_response_id`, adapters, diffusion models, and
custom `model_file` architectures remain unsupported. Other modality endpoints remain local-only.

DFlash/DSpark and MTP drafts can be selected through
[`POST /v1/cluster/start`](cluster-api.md#speculative-decoding). Both target and
draft checkpoints are staged on every worker before loading; draft weights remain
replicated. Worker CLI/environment draft settings are still rejected. Use the
cluster start contract to configure or disable drafting for the whole group.

Workers disable automatic prefix caching, vision caching, and the expiring
chat-compaction cache. These independent caches could otherwise skip work on
one rank. Normal local requests retain their existing serving behavior.
For multimodal requests, inline data URLs ensure each worker receives the same
media bytes; local media paths must exist identically on all workers.

For compaction, configure `compaction_key_file` in `cluster.json` and set
`MLX_VLM_COMPACTION_KEY_FILE` on every worker to a file containing the same Fernet
key. Keep the key across restarts. The coordinator compares decrypted capsule
contents, since encryption produces different ciphertext on each rank.

The coordinator returns the primary worker's response after comparing rank
outputs, excluding generated identifiers and timing metadata. A failed rank,
inference timeout, or disagreement writes `.cluster-failed.json` in the staging
directory and blocks further distributed requests. Local requests remain available.
An identity-poll timeout marks node telemetry stale (`telemetry_timeout`) and is
retried without failing the group. Connection failures or identity mismatches
still fail the group; successful later polling does not clear that failure.

Inspect that marker and worker logs, stop/restart the **entire affected worker
group**, and confirm no old request is running. The cluster start API clears a
failure marker only after authenticating fresh worker process identities. For
legacy manual recovery, only then remove the failure marker and retry. Do not delete an active lock file or restart unrelated groups.
Transfer failures occur before inference and can be retried; corrupt committed
snapshots are preserved for diagnosis instead of silently overwritten.

## Validation

```bash
python -m pytest -q mlx_vlm/tests/test_server_distributed.py \
  mlx_vlm/tests/test_server_distributed_launch.py mlx_vlm/tests/test_server_cluster.py \
  mlx_vlm/tests/test_server.py
```

Launcher tests cover startup/shutdown integration, ownership conflicts, occupied
endpoints, launch identity/backend mismatches, startup rollback, and unexpected
exit. They also run real supervisors (including a child ignoring SIGTERM) and
the installed `mlx.launch` with two local test workers. These tests do not
establish remote SSH or three-Mac JACCL operation.

The distributed tests cover HTTP streaming, cancellation, overlapping model
leases, staging integrity, input consistency, and failure handling. Two local
MLX ring processes also exercise the native continuous batch generator at batch
sizes 4 and 8, using a deterministic synthetic model with tensor collectives and
KV caches. They test reversed arrival order, late admission, early completion,
cancellation from only one rank, and sampling with deliberately different
process RNG states. This does not establish three-Mac JACCL
performance or released-weight model correctness; validate those separately on
the configured hardware.
