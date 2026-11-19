"""Compatibility imports for the shared native-reduction dense kernel."""

from ..wide_linear import _HEADER, _wide_kernel, native_wide_linear

__all__ = ["_HEADER", "_wide_kernel", "native_wide_linear"]
