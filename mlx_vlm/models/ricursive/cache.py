# Copyright © 2026 Neywa Labs Inc.

"""Per-request append positions for Ripple's full and sliding attention."""

import mlx.core as mx

from .. import cache as base


def _append_block(cache, keys, values, mask):
    length = keys.shape[2]
    keys, values = cache.update_and_fetch(keys, values)
    shape = (keys.shape[0], length, *keys.shape[1:])
    return (
        mx.broadcast_to(keys[:, None], shape),
        mx.broadcast_to(values[:, None], shape),
        mask.transpose(0, 2, 1, 3)[:, :, :, None],
    )


def _pad_single(cache, left_padding):
    if len(left_padding) != 1:
        raise ValueError("A populated cache must contain one request to batch")
    padding = left_padding[0]
    pad = [(0, 0), (0, 0), (padding, 0), (0, 0)]
    cache.keys, cache.values = mx.pad(cache.keys, pad), mx.pad(cache.values, pad)
    cache.left_padding += padding
    cache._idx += padding
    if hasattr(cache, "_offset"):
        cache._offset += padding
    return cache


class KVCache(base.KVCache):
    def fetch_verification_block(self, keys, values):
        length = keys.shape[2]
        mask = (
            mx.arange(self.offset + length) <= self.offset + mx.arange(length)[:, None]
        )
        mask = mx.broadcast_to(mask, (keys.shape[0], 1, *mask.shape))
        return _append_block(self, keys, values, mask)

    def to_batch(self, left_padding):
        if self.empty():
            return BatchKVCache(left_padding)
        return _pad_single(BatchKVCache.merge([self]), left_padding)

    @classmethod
    def merge(cls, caches):
        return BatchKVCache.merge(caches)

    def prefix_cache_merge(self, rows, prefix_lens):
        return self.merge(rows)


class BatchKVCache(base.BatchKVCache):
    """Roll back a row's cursor without relocating its accepted keys."""

    def __init__(self, left_padding):
        super().__init__(left_padding)
        self._ragged = False

    def update_and_fetch(self, keys, values):
        if not self._ragged:
            return super().update_and_fetch(keys, values)
        length = keys.shape[2]
        end = self._idx + length
        if end > self.keys.shape[2]:
            capacity = ((end + self.step - 1) // self.step) * self.step
            pad = [(0, 0), (0, 0), (0, capacity - self.keys.shape[2]), (0, 0)]
            self.keys, self.values = mx.pad(self.keys, pad), mx.pad(self.values, pad)
        positions = (self.offset + self.left_padding)[:, None] + mx.arange(length)
        positions = positions[:, None, :, None]
        self.keys = mx.put_along_axis(self.keys, positions, keys, axis=2)
        self.values = mx.put_along_axis(self.values, positions, values, axis=2)
        self.offset += length
        self._idx = end
        return self.keys[..., :end, :], self.values[..., :end, :]

    def make_mask(self, N, **kwargs):
        if not self._ragged:
            return super().make_mask(N, **kwargs)
        rows = (self.offset + self.left_padding)[:, None] + mx.arange(N)
        columns = mx.arange(self._idx + N)
        return (
            (columns <= rows[..., None]) & (columns >= self.left_padding[:, None, None])
        )[:, None]

    def fetch_verification_block(self, keys, values):
        return _append_block(self, keys, values, self.make_mask(keys.shape[2]))

    def trim_right(self, padding):
        self.offset -= mx.array(padding, mx.int32)
        self._ragged = True

    def extend(self, other):
        super().extend(other)
        self._ragged = self._ragged or other._ragged

    def extract(self, idx):
        result = KVCache()
        if idx < 0:
            idx += self.batch_size
        if not 0 <= idx < self.batch_size:
            raise IndexError("BatchKVCache row index out of range")
        if self.keys is not None:
            start, length = self.left_padding[idx].item(), self.offset[idx].item()
            result.keys = mx.contiguous(
                self.keys[idx : idx + 1, :, start : start + length]
            )
            result.values = mx.contiguous(
                self.values[idx : idx + 1, :, start : start + length]
            )
            result.offset = length
        return result

    @classmethod
    def merge(cls, caches):
        result = cls([0] * len(caches))
        result.__dict__.update(base.BatchKVCache.merge(caches).__dict__)
        return result

    @base.BatchKVCache.state.setter
    def state(self, value):
        base.BatchKVCache.state.fset(self, value)
        self._ragged = self.keys is not None
        self._right_padding = None


class _RingTransactions:
    @property
    def is_speculating(self):
        return getattr(self, "_speculation", None) is not None

    def start_speculation(self, length):
        if self.is_speculating:
            raise RuntimeError("A cache transaction is already active")
        self._generation = getattr(self, "_generation", 0) + 1
        self._speculation = None
        original = {
            name: mx.array(value) if isinstance(value, mx.array) else value
            for name, value in self.__dict__.items()
        }
        self._speculation = dict(
            original=original, updates=[], length=length, generation=self._generation
        )
        return self._generation

    def validate_speculation(self, lengths, generation=None):
        transaction = self._speculation
        if transaction is None or generation != transaction["generation"]:
            raise RuntimeError("Stale cache transaction")
        if len(lengths) not in (1, self.batch_size) or any(
            n < 0 or n > transaction["length"] for n in lengths
        ):
            raise ValueError("Invalid per-row accepted lengths")
        return transaction

    def abort_speculation(self, generation=None):
        if not self.is_speculating:
            return
        transaction = self.validate_speculation([0] * self.batch_size, generation)
        self.__dict__.update(transaction["original"])

    def commit_speculation(self, lengths, generation=None):
        transaction = self.validate_speculation(lengths, generation)
        lengths = list(lengths) * self.batch_size if len(lengths) == 1 else lengths
        if all(n == transaction["length"] for n in lengths):
            self._speculation = None
            return
        updates = transaction["updates"]
        self.__dict__.update(transaction["original"])
        if not updates or not max(lengths):
            return
        k, v = (mx.concatenate(parts, axis=2) for parts in zip(*updates))
        retained = max(lengths)
        self._append_verified(k[..., :retained, :], v[..., :retained, :], lengths)


class RotatingKVCache(_RingTransactions, base.RotatingKVCache):
    @property
    def batch_size(self):
        return 1 if self.keys is None else self.keys.shape[0]

    def update_and_fetch(self, keys, values):
        if self.is_speculating:
            self._speculation["updates"].append((keys, values))
        return super().update_and_fetch(keys, values)

    def _batch_view(self, batch_size):
        cache = BatchRotatingKVCache(self.max_size, [0] * batch_size)
        cache.keys, cache.values = self.keys, self.values
        cache.offset = mx.full((batch_size,), self.offset, mx.int32)
        cache._offset, cache._idx = self.offset, self._idx
        cache.rotated = self.offset > self.max_size
        return cache

    def _adopt(self, cache, length):
        self.keys, self.values = cache.keys, cache.values
        self.offset += length
        self._idx = (min(self._idx, self.max_size) + length) % self.max_size

    def fetch_verification_block(self, keys, values):
        if self.is_speculating:
            self._speculation["updates"].append((keys, values))
        cache = self._batch_view(keys.shape[0])
        result = cache.fetch_verification_block(keys, values)
        self._adopt(cache, keys.shape[2])
        return result

    def validate_speculation(self, lengths, generation=None):
        transaction = super().validate_speculation(lengths, generation)
        if len(set(lengths)) > 1:
            raise ValueError("Use a batched cache for independent accepted lengths")
        return transaction

    def _append_verified(self, keys, values, lengths):
        cache = self._batch_view(keys.shape[0])
        cache._write(keys, values, mx.array(lengths, mx.int32))
        self._adopt(cache, lengths[0])

    def to_batch(self, left_padding):
        if self.empty():
            return BatchRotatingKVCache(self.max_size, left_padding)
        return _pad_single(BatchRotatingKVCache.merge([self]), left_padding)

    @classmethod
    def merge(cls, caches):
        return BatchRotatingKVCache.merge(caches)

    def prefix_cache_merge(self, rows, prefix_lens):
        return self.merge(rows)


class BatchRotatingKVCache(_RingTransactions, base.BatchRotatingKVCache):
    """Independent ring cursors and transactional writes for each request."""

    def __init__(self, max_size, left_padding):
        super().__init__(max_size, left_padding)
        self._row_idx = None
        self._speculation = None
        self._generation = 0

    def _ring(self):
        if self.keys.shape[2] > self.max_size:
            trim = self.keys.shape[2] - self.max_size
            self.keys = self.keys[..., trim:, :]
            self.values = self.values[..., trim:, :]
            self._idx = self.max_size
            self.left_padding -= trim
        if self.keys.shape[2] < self.max_size:
            pad = [(0, 0), (0, 0), (0, self.max_size - self.keys.shape[2]), (0, 0)]
            self.keys, self.values = mx.pad(self.keys, pad), mx.pad(self.values, pad)
        if self._row_idx is None:
            self._row_idx = mx.full(
                self.offset.shape, self._idx % self.max_size, mx.int32
            )

    def _write(self, keys, values, lengths):
        if self.keys is None:
            self.keys = mx.zeros(
                (*keys.shape[:2], self.max_size, keys.shape[-1]), keys.dtype
            )
            self.values = mx.zeros(
                (*values.shape[:2], self.max_size, values.shape[-1]), values.dtype
            )
        self._ring()
        count = keys.shape[2]
        # Only the last window can survive a long appended segment.
        tokens = mx.maximum(lengths - self.max_size, 0)[:, None] + mx.arange(
            min(count, self.max_size)
        )
        positions = (self._row_idx[:, None] + tokens) % self.max_size
        indices = positions[:, None, :, None]
        active = (tokens < lengths[:, None])[:, None, :, None]
        for name, incoming in (("keys", keys), ("values", values)):
            old = getattr(self, name)
            update = mx.where(
                active,
                mx.take_along_axis(incoming, tokens[:, None, :, None], axis=2),
                mx.take_along_axis(old, indices, axis=2),
            )
            setattr(self, name, mx.put_along_axis(old, indices, update, axis=2))
        self.offset += lengths
        self._row_idx = (self._row_idx + lengths) % self.max_size
        self._offset += count
        self._idx = (self._idx + count) % self.max_size
        self.rotated = self._offset >= self.max_size

    def update_and_fetch(self, keys, values):
        if self._speculation is not None:
            self._speculation["updates"].append((keys, values))
        if self._row_idx is None:
            return super().update_and_fetch(keys, values)
        length = keys.shape[2]
        if length > 1:
            order = (self._row_idx[:, None] + mx.arange(self.max_size)) % self.max_size
            context = [
                mx.take_along_axis(x, order[:, None, :, None], axis=2)
                for x in (self.keys, self.values)
            ]
            result = tuple(
                mx.concatenate([old, new], axis=2)
                for old, new in zip(context, (keys, values))
            )
        self._write(keys, values, mx.full(self.offset.shape, length, mx.int32))
        if length > 1:
            return result
        width = min(self._offset, self.max_size)
        return self.keys[..., :width, :], self.values[..., :width, :]

    def _append_verified(self, keys, values, lengths):
        self._write(keys, values, mx.array(lengths, mx.int32))

    def trim(self, n):
        if self._row_idx is None:
            return super().trim(n)
        n = min(n, self._offset)
        self._offset -= n
        self._idx = (self._idx - n) % self.max_size
        self.offset -= n
        self._row_idx = (self._row_idx - n) % self.max_size
        return n

    def fetch_verification_block(self, keys, values):
        """Gather causal ring views and append the complete block once."""
        length = keys.shape[2]
        if self.keys is None:
            self.keys = mx.zeros(
                (*keys.shape[:2], self.max_size, keys.shape[-1]), keys.dtype
            )
            self.values = mx.zeros_like(self.keys)
        self._ring()
        width = min(self._offset + length, self.max_size)
        queries = mx.arange(length)[:, None]
        tokens = mx.maximum(queries - self.max_size + 1, 0) + mx.arange(
            min(length, self.max_size)
        )
        positions = (self._row_idx[:, None, None] + tokens) % self.max_size
        indices = positions[:, :, None, :, None]
        causal = (tokens <= queries)[None, :, None, :, None]
        views = []
        for old, new in ((self.keys, keys), (self.values, values)):
            incoming = mx.take_along_axis(
                new[:, None], tokens[None, :, None, :, None], axis=3
            )
            previous = mx.take_along_axis(old[:, None], indices, axis=3)
            expanded = mx.broadcast_to(
                old[:, None], (old.shape[0], length, *old.shape[1:])
            )
            view = mx.put_along_axis(
                expanded, indices, mx.where(causal, incoming, previous), axis=3
            )
            views.append(view[..., :width, :])
        age = (
            self._row_idx[:, None, None] + queries[None] - mx.arange(width)
        ) % self.max_size
        mask = age < mx.minimum(
            self.offset[:, None, None] + queries[None] + 1, self.max_size
        )
        if self.is_speculating:
            self._speculation["updates"].append((keys, values))
        self._write(keys, values, mx.full(self.offset.shape, length, mx.int32))
        return *views, mask[:, :, None, None, :]

    def make_mask(self, N, window_size=None, return_array=False):
        if self._row_idx is None:
            return super().make_mask(N, window_size, return_array)
        window = window_size or self.max_size
        if N == 1:
            width = min(self._offset + N, self.max_size)
            age = (self._row_idx[:, None] - mx.arange(width)) % self.max_size
            return (age < mx.minimum(self.offset + 1, window)[:, None])[:, None, None]
        rows = self.max_size + mx.arange(N)
        columns = mx.arange(self.max_size + N)
        return ((columns <= rows[:, None]) & (columns > rows[:, None] - window))[
            None, None
        ] & (columns >= (self.max_size - self.offset)[:, None, None, None])

    def filter(self, indices):
        super().filter(indices)
        if self._row_idx is not None:
            self._row_idx = self._row_idx[indices]

    def _temporal_order(self):
        if self._row_idx is None:
            return super()._temporal_order()
        positions = (self._row_idx[:, None] + mx.arange(self.max_size)) % self.max_size
        self.keys = mx.take_along_axis(self.keys, positions[:, None, :, None], axis=2)
        self.values = mx.take_along_axis(
            self.values, positions[:, None, :, None], axis=2
        )
        self.left_padding = self.max_size - mx.clip(self.offset, 0, self.max_size)
        self._idx = self._offset = self.max_size
        self.rotated = False
        self._row_idx = None

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if self._row_idx is not None and right_padding is not None:
            self._temporal_order()
        return super().prepare(
            left_padding=left_padding, lengths=lengths, right_padding=right_padding
        )

    def extend(self, other):
        if self._row_idx is None and other._row_idx is None:
            return super().extend(other)
        self._ring()
        other._ring()
        for name in ("keys", "values", "offset", "left_padding", "_row_idx"):
            setattr(
                self, name, mx.concatenate([getattr(self, name), getattr(other, name)])
            )
        self._offset = max(self._offset, other._offset)

    def extract(self, idx):
        if self._row_idx is None:
            result = super().extract(idx)
        else:
            result = base.RotatingKVCache(self.max_size)
            length = min(self.max_size, self.offset[idx].item())
            positions = (
                self._row_idx[idx] - length + mx.arange(length)
            ) % self.max_size
            result.keys = mx.take(self.keys[idx : idx + 1], positions, axis=2)
            result.values = mx.take(self.values[idx : idx + 1], positions, axis=2)
            result.offset = self.offset[idx].item()
            result._idx = length
        wrapped = RotatingKVCache(self.max_size)
        wrapped.__dict__.update(result.__dict__)
        return wrapped

    @classmethod
    def merge(cls, caches):
        result = cls(caches[0].max_size, [0] * len(caches))
        result.__dict__.update(base.BatchRotatingKVCache.merge(caches).__dict__)
        return result

    @property
    def state(self):
        if self.keys is None:
            return None, None, self.offset, self.left_padding, self._row_idx
        return self.keys, self.values, self.offset, self.left_padding, self._row_idx

    @state.setter
    def state(self, value):
        self.keys, self.values, self.offset, self.left_padding, self._row_idx = value
        self._speculation, self._generation = None, 0
        self._lengths = None

    @base.BatchRotatingKVCache.meta_state.setter
    def meta_state(self, value):
        self.max_size, self._offset, self._idx = map(int, value[:3])
        self.rotated = value[3] in (True, "True", "1")
