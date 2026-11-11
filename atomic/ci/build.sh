#!/usr/bin/env bash
# Freeze mlx-server into one binary with PyInstaller. Run it from the root of the upstream tag with
# atomic/patches applied and the package installed (`pip install pyinstaller .`); the binary is
# dist/mlx-server. Signing and notarization are Release (Atomic)'s job, not this script's.
#
# The install must not be editable (-e): PyInstaller then copies mlx_vlm as source without analyzing
# its imports, and a dependency only mlx_vlm imports is left out. At v0.7.12 that was
# cryptography.fernet (server/compaction.py), and the binary did not start.
#
# --onefile keeps drop-in compatibility with what Atomic Chat's core unpacks: a single mlx-server
# beside nothing else. mlx-vlm resolves model, drafter and tool-parser modules lazily through
# importlib at request time, so those packages are collected whole; a model family upstream adds
# lands in the binary without a change here. A new lazily imported third-party dependency does not:
# when a smoke or a request fails with ModuleNotFoundError in the frozen build, add it below.
#
# llguidance is --collect-all, not --hidden-import: its __init__ calls
# importlib.metadata.version("llguidance"), so a bundle with the modules but without the dist-info
# raises PackageNotFoundError and structured response_format requests silently fall back to
# unconstrained generation. smoke.py's --selftest-structured catches that.
set -euo pipefail

ci="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pyinstaller \
  --noconfirm \
  --onefile \
  --name mlx-server \
  --collect-all mlx \
  --collect-all mlx_vlm \
  --collect-all mlx_audio \
  --collect-submodules mlx_vlm.models \
  --collect-submodules mlx_vlm.speculative \
  --collect-submodules mlx_vlm.tool_parsers \
  --collect-data transformers \
  --collect-data tokenizers \
  --collect-all llguidance \
  --hidden-import fastapi \
  --hidden-import uvicorn \
  --hidden-import uvicorn.logging \
  --hidden-import uvicorn.loops \
  --hidden-import uvicorn.loops.auto \
  --hidden-import uvicorn.protocols \
  --hidden-import uvicorn.protocols.http \
  --hidden-import uvicorn.protocols.http.auto \
  --hidden-import uvicorn.protocols.websockets \
  --hidden-import uvicorn.protocols.websockets.auto \
  --hidden-import uvicorn.lifespan \
  --hidden-import uvicorn.lifespan.on \
  --hidden-import huggingface_hub \
  --hidden-import miniaudio \
  --hidden-import cv2 \
  --exclude-module gradio \
  --exclude-module jupyter \
  --exclude-module IPython \
  "$ci/mlx_server_entry.py"

file dist/mlx-server
ls -lh dist/mlx-server
