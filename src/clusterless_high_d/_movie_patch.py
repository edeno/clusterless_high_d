"""Workaround for non_local_detector #42.

``make_single_environment_movie`` sets the multiunit-panel axis limits only on
global ``frame_idx == 0``. Under the default 4-worker parallel render, every
worker but the one owning the first chunk leaves that panel at default limits,
so the firing-rate line is drawn off-screen and "disappears" after the first
chunk boundary.

This module provides a drop-in render function that calls the original renderer
and then (re)sets the bottom-panel limits on *every* frame, so the line stays
visible in all chunks while preserving parallelism. ``render_frame_with_fixed_axes``
is a module-level function so it remains picklable for the worker processes.

Usage
-----
>>> from clusterless_high_d import _movie_patch
>>> _movie_patch.apply()   # before calling make_single_environment_movie
"""

from __future__ import annotations

import numpy as np
from non_local_detector.visualization import movie as _movie

# Captured at import, before apply() patches the module global, so it is always
# the genuine original (no self-recursion).
_ORIGINAL_RENDER = _movie._render_single_env_frame


def render_frame_with_fixed_axes(fig, axes, frame_idx, data):
    """Original frame render + per-frame bottom-panel limit setting."""
    _ORIGINAL_RENDER(fig, axes, frame_idx, data)
    window_ind = data["window_ind"]
    sampling_frequency = data["sampling_frequency"]
    rate_max = float(np.max(data["rate"])) if len(data["rate"]) else 1.0
    axes["ax1"].set_ylim(0.0, rate_max if rate_max > 0 else 1.0)
    axes["ax1"].set_xlim(
        window_ind[0] / sampling_frequency,
        window_ind[-1] / sampling_frequency,
    )


def apply() -> None:
    """Patch movie._render_single_env_frame to the fixed version."""
    _movie._render_single_env_frame = render_frame_with_fixed_axes
