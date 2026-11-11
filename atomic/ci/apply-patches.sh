#!/usr/bin/env bash
# Apply atomic/patches/*.patch to the checked-out upstream tag, in name order, at the repository
# root (mlx-vlm is one Python package; there are no submodules).
#
# --check changes nothing. Each patch goes through `git apply --check` against a scratch index that
# already holds the patches before it, as the real run would apply them; every one that does not
# apply is reported with git's reason, and the exit status is 1 if any did not. Release (Atomic)'s
# `patches` job runs this, and the hatchet pipeline reads a failure there as a patch conflict.
set -euo pipefail
shopt -s nullglob

check=0
case "${1:-}" in
  --check) check=1 ;;
  "") ;;
  *) echo "usage: $0 [--check]" >&2; exit 2 ;;
esac

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../patches" && pwd)"

if [ "$check" = 1 ]; then
  scratch="$(mktemp -d)"
  trap 'rm -rf "$scratch"' EXIT
  index="$scratch/root.index"
  target=.
  GIT_INDEX_FILE="$index" git read-tree HEAD
  total=0
  failed=""
  for patch in "$dir"/*.patch; do
    name="$(basename "$patch")"
    total=$((total + 1))
    if out="$(GIT_INDEX_FILE="$index" git -C "$target" apply --check --cached --verbose --whitespace=nowarn "$patch" 2>&1)"; then
      echo "::group::$name applies in $target"
      echo "$out"
      echo "::endgroup::"
      # Into the scratch index only, so the next patch is checked on top of this one.
      GIT_INDEX_FILE="$index" git -C "$target" apply --cached --whitespace=nowarn "$patch"
      continue
    fi
    # Outside a group, so git's reason is not folded away in the job log.
    echo "$name does not apply in $target:"
    echo "$out"
    failed="$failed $name"
    # A patch that landed upstream no longer applies but reverses cleanly: it is to be deleted.
    if GIT_INDEX_FILE="$index" git -C "$target" apply --check --cached --reverse --whitespace=nowarn "$patch" 2>/dev/null; then
      echo "::error title=Patch already upstream::$name is already in the tree (it reverses cleanly): delete it"
    else
      echo "::error title=Patch does not apply::$name does not apply in $target"
    fi
  done
  if [ -n "$failed" ]; then
    echo "$(echo "$failed" | wc -w | tr -d ' ') of $total patch(es) do not apply:$failed"
    exit 1
  fi
  echo "all $total patch(es) apply"
  exit 0
fi

applied=0
for patch in "$dir"/*.patch; do
  name="$(basename "$patch")"
  echo "::group::git apply $name"
  git apply --verbose --whitespace=nowarn "$patch"
  echo "::endgroup::"
  applied=$((applied + 1))
done
echo "applied $applied patch(es)"
