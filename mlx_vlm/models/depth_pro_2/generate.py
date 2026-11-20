"""Predict metric depth for photos, stored videos, and live RGB streams."""

import math
from collections import deque
from functools import cached_property
from itertools import islice

import mlx.core as mx
import numpy as np

from .encoder import CrossFrameCache
from .focal import FOCAL_CHECKPOINT, FocalLengthEstimator, focal_length_from_fov
from .processing_depth_pro_2 import DepthPro2Processor

DEFAULT_WINDOW = 16
DEFAULT_OVERLAP = 4


def read_image(source):
    """Read a local file, URL, or PIL image as uint8 RGB pixels."""
    from ...utils import load_image

    return np.asarray(load_image(source))


def prediction(inverse_depth, focal_length):
    return {
        "inverse_depth": inverse_depth,
        "depth": (
            None
            if focal_length is None
            else focal_length / (inverse_depth.shape[-1] * inverse_depth)
        ),
        "focallength_px": focal_length,
    }


class DepthPro2Predictor:
    """Depth extraction using Apple's depth weights and optional focal estimator."""

    def __init__(self, model, processor=None, *, focal_checkpoint=FOCAL_CHECKPOINT):
        self.model = model
        self.processor = processor if processor is not None else DepthPro2Processor()
        self.focal_checkpoint = focal_checkpoint

    @cached_property
    def focal_model(self):
        if self.model.fov is not None:
            return self.model.fov
        if self.focal_checkpoint is not None:
            return FocalLengthEstimator.from_pretrained(self.focal_checkpoint)
        return None

    def infer(self, image, *, f_px=None):
        """Predict depth from an RGB image or a short clip attended jointly."""
        frames = self.processor(image)["pixel_values"]
        single = frames.ndim == 3
        if single:
            frames = frames[None]
        focal = self._focal_argument(f_px)
        if focal is None and self.focal_model is not None:
            focal = self.focal_model.focal_length_px(frames)
        inverse_depth = self.model(frames)
        return prediction(inverse_depth[0] if single else inverse_depth, focal)

    @staticmethod
    def _focal_argument(f_px):
        if f_px is None:
            return None
        focal = float(f_px)
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError("Focal length must be positive and finite")
        return mx.array(focal, dtype=mx.float32)

    def stream(
        self,
        window=DEFAULT_WINDOW,
        overlap=DEFAULT_OVERLAP,
        f_px=None,
        *,
        mode="buffered",
    ):
        """Open a stream whose step() and flush() return completed frame predictions."""
        return DepthStream(self, window, overlap, self._focal_argument(f_px), mode)

    def infer_video(
        self,
        frames,
        window=DEFAULT_WINDOW,
        overlap=DEFAULT_OVERLAP,
        f_px=None,
        *,
        mode="buffered",
    ):
        """Yield one prediction per RGB frame using the first window's focal estimate."""
        stream = self.stream(window, overlap, f_px, mode=mode)
        frames = iter(frames)
        if f_px is None:
            first = [stream._prepare_frame(frame) for frame in islice(frames, 1)]
            if not first:
                return
            estimator = self.focal_model
            if estimator is not None:
                first.extend(
                    stream._prepare_frame(frame) for frame in islice(frames, window - 1)
                )
                stream.focal_length = estimator.focal_length_px(mx.stack(first))
                mx.eval(stream.focal_length)
            for frame in first:
                yield from stream._add_frame(frame)
            del first
        for frame in frames:
            yield from stream.step(frame)
        yield from stream.flush()


class DepthStream:
    def __init__(self, predictor, window, overlap, focal_length, mode):
        if mode not in ("buffered", "unbuffered"):
            raise ValueError("Mode must be 'buffered' or 'unbuffered'")
        if window < 1:
            raise ValueError("Window must contain at least one frame")
        if mode == "buffered" and not 0 <= overlap < window:
            raise ValueError("Overlap must be nonnegative and smaller than the window")
        self.predictor = predictor
        self.window = window
        self.overlap = overlap if mode == "buffered" else 0
        self.cache = CrossFrameCache(window) if mode == "unbuffered" else None
        self.clip_size = 1 if self.cache is not None else window
        self.frames = deque(maxlen=self.clip_size)
        self.pending = []
        self.new_frames = 0
        self.focal_length = focal_length
        self.fov_history = []
        self.closed = False

    def _prepare_frame(self, frame):
        pixels = self.predictor.processor(frame)["pixel_values"]
        if pixels.ndim != 3:
            raise ValueError("Depth Pro 2 stream expects one RGB frame at a time")
        return pixels

    def step(self, frame):
        return self._add_frame(self._prepare_frame(frame))

    def _add_frame(self, frame):
        if self.closed:
            raise RuntimeError("Open a new Depth Pro 2 stream after flush()")
        if self.frames and frame.shape != self.frames[-1].shape:
            raise ValueError(
                "All frames in a Depth Pro 2 stream must have the same shape"
            )
        self.frames.append(frame)
        self.new_frames += 1
        if len(self.pending) + self.new_frames < self.clip_size:
            return []
        return self._predict(len(self.frames), keep=self.overlap)

    def flush(self):
        if self.closed:
            return []
        if self.new_frames:
            count = len(self.frames) if self.overlap else self.new_frames
            done = self._predict(count, keep=0)
        else:
            done = self._emit(len(self.pending), self.focal_length)
        self.closed = True
        self.frames.clear()
        self.fov_history.clear()
        self.cache = None
        return done

    def _predict(self, length, keep):
        pixels = mx.stack(list(self.frames)[-length:])
        inverse = self.predictor.model(pixels, cache=self.cache)
        if self.cache is not None:
            self.cache.materialize(inverse)
        else:
            mx.eval(inverse)
        focal = self._estimate_focal(pixels)
        held = len(self.pending)
        inverse = inverse[length - held - self.new_frames :]
        self.new_frames = 0
        if held:
            ramp = mx.linspace(1.0, 0.0, held + 2)[1:-1, None, None]
            blended = ramp * mx.stack(self.pending) + (1 - ramp) * inverse[:held]
            mx.eval(blended)
            self.pending = list(blended)
        self.pending.extend(inverse[held:])
        return self._emit(len(self.pending) - keep, focal)

    def _estimate_focal(self, frames):
        if self.focal_length is not None or self.predictor.focal_model is None:
            return self.focal_length
        angles = self.predictor.focal_model(frames)
        mx.eval(angles)
        self.fov_history.append(angles)
        angles = mx.concatenate(self.fov_history)
        focal = focal_length_from_fov(angles, *frames.shape[1:3])
        if len(angles) >= self.window:
            self.focal_length = focal
            self.fov_history.clear()
        return focal

    def _emit(self, count, focal):
        done, self.pending = self.pending[:count], self.pending[count:]
        return [prediction(inverse, focal) for inverse in done]
