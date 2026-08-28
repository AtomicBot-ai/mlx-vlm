"""Thin entrypoint for PyInstaller --onefile builds of the MLX server.

Re-exports `mlx_vlm.server:main` so PyInstaller has a single, predictable
script to freeze (`mlx_vlm/_entry_server.py`) without having to navigate
the package's import-time side effects.

Used by `.github/workflows/build-mlxvlm-macos.yml` to produce the
`mlx-server` binary shipped with Atomic-Chat.
"""

import sys

from mlx_vlm.server import main

SELFTEST_FLAG = "--selftest-structured"


def _selftest_structured() -> int:
    """Assert the frozen bundle can actually constrain generation.

    The structured-output stack is optional and only loads at *request* time,
    so a bundle missing it answers 400 and silently degrades every
    `response_format` request to unconstrained generation instead of failing
    the build. Shipping the llguidance modules is not enough: llguidance reads
    its own `importlib.metadata.version` on import, so the dist-info is
    load-bearing too. This walks the exact import path the server takes.
    """
    try:
        import llguidance
        import llguidance.hf  # noqa: F401
        import llguidance.numpy  # noqa: F401

        from mlx_vlm.structured import LLGuidanceLogitsProcessor  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - reporting the cause is the point
        print(f"selftest FAILED: structured output unavailable: {exc!r}")
        return 1
    print(f"selftest OK: llguidance {llguidance.__version__}")
    return 0


if __name__ == "__main__":
    if SELFTEST_FLAG in sys.argv[1:]:
        raise SystemExit(_selftest_structured())
    main()
