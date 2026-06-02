"""Data loading module for clusterless high-d decoding.

Loader functions require a Spyglass/DataJoint connection. They are
exposed via PEP 562 ``__getattr__`` so that importing this package
does not open a connection unless a loader is actually called.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_LAZY_EXPORTS = {
    "load_position": "clusterless_high_d.data_loaders.position",
    "load_spikes_by_area": "clusterless_high_d.data_loaders.spikes",
    "get_sort_group_brain_areas": "clusterless_high_d.data_loaders.utils",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


if TYPE_CHECKING:
    from .position import load_position  # noqa: F401
    from .spikes import load_spikes_by_area  # noqa: F401
    from .utils import get_sort_group_brain_areas  # noqa: F401
