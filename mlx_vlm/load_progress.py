# Copyright © 2026 Neywa Labs Inc.
"""Measured loading progress shared by library callers and serving workers."""

import time
from typing import Callable, Optional

import mlx.core as mx

ProgressCallback = Callable[[dict], None]


class LoadingProgress:
    def __init__(self, model, callback: Optional[ProgressCallback] = None):
        self.model = str(model)
        self.callback = callback
        self.started = time.monotonic()
        self.snapshot = None

    def __call__(self, stage, **counts):
        self.update({"stage": stage, **counts})

    def update(self, event):
        unit = {
            "loading_checkpoint": "files",
            "evaluating_layers": "layers",
        }.get(event["stage"])
        completed = event.get(f"completed_{unit}")
        total = event.get(f"total_{unit}")
        # A stage percentage is a count ratio, never an estimate of total load time.
        stage_percent = (
            round(100 * completed / total, 2)
            if total and completed is not None
            else None
        )
        # Publish a new snapshot; observers may read it from another thread.
        self.snapshot = {
            **event,
            "stage_percent": stage_percent,
            "model": self.model,
            "elapsed_seconds": time.monotonic() - self.started,
            "updated_at": time.time(),
        }
        if self.callback is not None:
            self.callback(self.snapshot.copy())


def evaluate_model(model, progress_callback=None, *, clear_cache=False):
    """Materialize parameters, counting layers only after their evaluation ends.

    Without an observer, preserve the ordinary whole-model evaluation path.
    Remaining parameters include any vision/audio/projector modules.
    """
    if progress_callback is None:
        mx.eval(model.parameters())
        return

    language_model = getattr(model, "language_model", model)
    layers = getattr(language_model, "layers", [])
    progress_callback(
        {
            "stage": "evaluating_layers",
            "completed_layers": 0,
            "total_layers": len(layers),
        }
    )
    for index, layer in enumerate(layers):
        mx.eval(layer.parameters())
        if clear_cache:
            mx.clear_cache()
        progress_callback(
            {
                "stage": "evaluating_layers",
                "completed_layers": index + 1,
                "total_layers": len(layers),
            }
        )
    progress_callback({"stage": "evaluating_remaining_parameters"})
    mx.eval(model.parameters())
