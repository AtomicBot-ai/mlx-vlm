# Atomic builds of mlx-vlm

This fork builds the `mlx-server` binary [Atomic Chat](https://github.com/AtomicBot-ai/Atomic-Chat)
runs on macOS arm64, from an unmodified [Neywa-Labs/mlx-vlm](https://github.com/Neywa-Labs/mlx-vlm)
release tag (`vX.Y.Z`; formerly Blaizzy/mlx-vlm, which redirects there) plus the patch series in
[`atomic/patches/`](patches/).

| Asset | What it is |
| --- | --- |
| `mlxvlm-mlx-server-macos-arm64.tar.gz` | `mlx-server`: a PyInstaller onefile, signed with the Developer ID (hardened runtime, `atomic/ci/mlx-server.entitlements`) and notarized |
| `requirements.lock.txt` | What pip resolved for that build: `requirements.txt` has only lower bounds |

Users get it through atomic-chat-conf: its `mirror-mlx.yml` copies the asset into a conf release
`mlxvlm-macos-arm64-<rev>` and writes `backends/mlx-manifest.staging.json`; the core's
`engine-candidate` check (`engine=mlx`) and conf's `mlx-approve.yml` move it to the live
`backends/mlx-manifest.json`. Released cores accept only a `tag_name` matching
`^mlxvlm-macos-arm64-[0-9a-f]{7,40}$`, which is why conf names its releases that way, and they never
go back to an older build (they order builds by `published_at`): a bad build is fixed by a newer one,
not by reverting the manifest.

## Branches

| Branch | What it is |
| --- | --- |
| `main` | A fast-forward mirror of upstream `main`. Never commit here. |
| `atomic` | The default branch: upstream plus **only added files** (`atomic/`, `.github/workflows/release-atomic.yml`). |
| `upstream-sync/<tag>` | `atomic` with an upstream tag merged in, built and tested by the hatchet pipeline, merged back into `atomic` through a PR when the core's check is green. |
| `legacy/main-v0.6.15` | The fork before the patch series: upstream v0.6.15+15 with our commits on top, built by `build-mlxvlm-macos.yml` into the `mlxvlm-macos-arm64-<sha7>` releases. Kept for those releases; do not build from it. |

Source code is never changed on a branch. Every build checks out the **upstream tag** and applies
the patch series, so a sync is a merge of upstream files only, never a conflict in our code, and the
release notes say exactly what differs.

## The pipeline

[atomic-hatchet-pipeline](https://github.com/AtomicBot-ai/atomic-hatchet-pipeline) watches upstream
releases (engine `mlx`). For a new tag it:

1. fast-forwards `main` and merges the tag into `upstream-sync/<tag>`;
2. **reviews the patches**: for every patch whose files upstream changed since the last synced tag,
   its agent decides to keep it, rewrite it for the new code, or delete it, citing the upstream
   commit that made it unnecessary; patches upstream did not touch are not reviewed;
3. dispatches `release-atomic.yml` on the branch (`publish`, `replace`); if a patch no longer
   applies or the build fails, its agent fixes `atomic/patches` or `atomic/ci` once, and a patch
   that reverses cleanly (already upstream) is deleted without the agent;
4. runs the core's `engine-candidate` check on the release, and on green rolls it out through conf.

The agent may not edit `atomic/tests/` or `atomic/ci/smoke.py`: they are what tells a correct fix
from a wrong one. When a review deletes a patch and upstream does not in fact do the same thing,
its test in `atomic/tests/test_patches.py` fails the build.

## Building a release by hand

1. Fast-forward `main` to upstream (`gh repo sync AtomicBot-ai/mlx-vlm -b main`, or the
   merge-upstream API), so the tag's commit exists here: the release is created at it.
2. Actions -> **Release (Atomic)** on `atomic` (or a sync branch: a dispatch on a branch runs that
   branch's copy of the workflow and its `atomic/`):
   - `upstream_tag`: e.g. `v0.7.12` (must match `^v\d+\.\d+\.\d+$`);
   - `publish`: creates the prerelease `<tag>` here, titled `mlx-server <tag> (Atomic)`. If that
     release exists, the run fails at once, unless
   - `replace` is on: the build's assets overwrite the release's;
   - `candidate_id`: goes into the run title as `[cid:<id>]`, which is how the pipeline finds its
     run; `manual` by default.

   Run it once with `publish` off first: the binary is built, signed, smoke-tested and notarized,
   nothing is published.
3. In atomic-chat-conf, mirror it: `<rev>` is the first seven characters of the commit the build
   ran at (the run's `GITHUB_SHA`, also in the release notes).

   ```bash
   gh workflow run mirror-mlx.yml -R AtomicBot-ai/atomic-chat-conf -f tag=v0.7.12 -f atomic_rev=<rev>
   ```

What the build checks, in order: every patch applies (`patches` job; keep its id and name: the
pipeline tells a patch conflict from a broken build by it), `mlx_vlm/version.py` matches the tag,
the Gemma 4 processor imports without torch, `atomic/tests` pass on the patched tree, and the signed
binary passes `atomic/ci/smoke.py`: `--help` prints `usage: mlx-server` and every flag the core
passes, `--selftest-structured` loads llguidance, and the server, started on
`mlx-community/SmolLM2-135M-Instruct-8bit` (pinned revision, cached per runner) with
`MLX_VLM_SINGLE_MODEL=1`, answers a chat completion that names another model and a `json_schema`
response format with JSON.

### By hand on a Mac

```bash
git fetch upstream --tags
git worktree add --detach ../mlx-vlm-v0.7.12 v0.7.12
cd ../mlx-vlm-v0.7.12
bash ../mlx-vlm/atomic/ci/apply-patches.sh --check   # what the patches job saw, changes nothing
bash ../mlx-vlm/atomic/ci/apply-patches.sh
python3.12 -m venv .venv && .venv/bin/pip install pyinstaller pytest .
.venv/bin/python -m pytest -q ../mlx-vlm/atomic/tests
PATH="$PWD/.venv/bin:$PATH" bash ../mlx-vlm/atomic/ci/build.sh
.venv/bin/python -I ../mlx-vlm/atomic/ci/smoke.py --binary dist/mlx-server --models-dir ~/mlx-smoke-models
```

Install the package with `pip install .`, never `-e`: PyInstaller does not analyze an editable
install's imports and leaves out dependencies only mlx-vlm imports (`atomic/ci/build.sh`).

### Changing a patch

Work in a worktree of the tag with the series applied, change the code, and write the diff of the
touched files back into the patch, keeping its header (`git diff -- <files>` after `git add -N` for
new files; or commit each patch and `git format-patch`, then put the header back). Every patch must
apply on top of the ones before it. A new behavior gets a test in `atomic/tests/test_patches.py`
that fails without its patch. Commit only `atomic/` or `release-atomic.yml` on `atomic`.

## Upstream workflows that must stay disabled

Upstream's workflow files come with every sync and stay in the tree (the branch only adds files), so
they are disabled in this fork's **Actions** settings:

| Workflow | Why |
| --- | --- |
| `python-publish.yml` | Publishes to PyPI on a release; every release here is ours. |
| `tests.yml` | Upstream's suite on every push to `main` (the mirror) and every pull request: runner time for nothing; `atomic/tests` and the smoke cover what we change. |
| `pr-contributor-reminder.yml` | Comments on pull requests (`pull_request_target`), including the pipeline's. |
| `build-mlxvlm-macos.yml` | The pre-patch build on `legacy/main-v0.6.15`; would release `mlxvlm-macos-arm64-<sha7>` from it. |

A workflow upstream adds later arrives enabled with the next sync: disable it unless it is harmless.

## Dropped when the fork moved to patches (v0.6.15 -> v0.7.12)

The fork's commits up to `legacy/main-v0.6.15` were checked against v0.7.12, Atomic Chat's core and
app, and the 37 MLX models of the conf catalog (2026-11-12). Only what those need was kept.

v0.7.12 does the same itself:

| Change | Why it is gone |
| --- | --- |
| MTP accepted counts as a list (`gemma4`, `deepseek_v4` `rollback_speculative_cache`) | Rollback now goes through `speculative/cache_state.py`, which takes ints, arrays and lists. |
| Text-only checkpoints of VLM wrappers (`_is_text_only_checkpoint`) | `load_model` passes the checkpoint manifest to `_drop_modules_without_weights`, which disables towers without weights. |
| Vestigial KV-shared weights of MLX-format Gemma 4 | `load_model` sanitizes every checkpoint, and Gemma 4's `sanitize` drops them. |
| `chat_template_kwargs.enable_thinking` | `ChatRequest` copies it to the top level when the request has none there. |
| `mlx-lm>=0.31.3`, `mlx-audio>=0.4.7` in requirements | Nothing imports mlx_lm since the models were ported (#1891); upstream requires `mlx-audio>=0.6.2`. |
| Gemma 4 Unified, video input, quantization predicate (cherry-picked #1267, #1280, #1288, #1292) | Merged upstream. |

Nothing Atomic Chat uses needs it:

| Change | Why it is gone |
| --- | --- |
| Stop tokens from `text_config.eos_token_id` and the tokenizer_config `eot_token` | Upstream stops on `generation_config.json`'s `eos_token_id`, which for every Gemma 4 in the catalog already holds `<turn|>` (106); the only catalog model it changed, `AtomicChat/Qwen3.5-9B-MLX-4bit`, gained `<\|endoftext\|>` while `<\|im_end\|>` already stops it. |
| Gemma 3 / Gemma 4 chat-template fallback | Every Gemma in the catalog ships its template; only broken third-party conversions needed it. |
| `thinking_budget`, `thinking_start_token`, `thinking_end_token` from `chat_template_kwargs` | For MLX the app sends `thinking_budget` at the top level, and the core puts only `enable_thinking` into the kwargs. |
| The llguidance import error in the structured-output message | Diagnostics only; the build's `--selftest-structured` catches a bundle without llguidance. |
| Comments in `server/openai.py`, Black formatting in `prompt_utils.py` | No behavior. |
