"""Thin entrypoint for PyInstaller --onefile builds of the MLX server.

Re-exports `mlx_vlm.server:main` so PyInstaller has a single, predictable
script to freeze (`mlx_vlm/_entry_server.py`) without having to navigate
the package's import-time side effects.

Used by `.github/workflows/build-mlxvlm-macos.yml` to produce the
`mlx-server` binary shipped with Atomic-Chat.
"""

from mlx_vlm.server import main

if __name__ == "__main__":
    main()
