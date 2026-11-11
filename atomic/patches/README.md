# Patch series

`*.patch` files here are applied to the checked-out upstream tag by `atomic/ci/apply-patches.sh`, in
name order, at the repository root. Name them `NNNN-short-description.patch`. Keep each patch small
and about one behavior, and write its header (the text above the first `diff --git`) for someone
deciding whether it is still needed:

- what upstream does and why that is not enough for Atomic Chat;
- what uses the behavior (the core, the app);
- **when to drop it** ("Drop it when upstream ...");
- which test in `atomic/tests/test_patches.py` guards it.

The hatchet pipeline's agent reads these headers when a new upstream tag touches a patched file: it
keeps the patch, rewrites it for the new code, or deletes it with the upstream commit that made it
unnecessary. Offer a patch upstream when you can: one that lands upstream is deleted here.

`apply-patches.sh --check`, run from the root of a checkout of the tag, changes nothing: it lists
every patch that does not apply on top of the ones before it, with git's reason, and flags the ones
that reverse cleanly as already upstream. A patch that only inserts lines can apply twice, so a
clean `--check` does not prove that upstream has not done the same; the tests do.

Current series (on v0.7.12):

| Patch | Why |
| --- | --- |
| `0001-server-single-model-mode.patch` | `MLX_VLM_SINGLE_MODEL=1` (set by the core) keeps serving the loaded model whatever label a request names, instead of unloading it and fetching the label from Hugging Face. The core sends its own model id; released cores get every build, so this stays until upstream has an equivalent. |
| `0002-server-honor-suppress-tokens.patch` | `generation_config.json` `suppress_tokens` become a default logit bias, so Gemma 4 omni's image/audio soft tokens never leak into replies (`AtomicChat/gemma-4-12B-it-MLX-4bit`). |

Both are worth offering upstream; when they land there, the series is empty and the fork builds
upstream tags as they are.
