"""Deferred CUDA-event profiling, gated by `SSD_PROFILE=1`.

Records events without per-call `torch.cuda.synchronize()` — sync happens once
at `flush()`. This avoids GPU drains perturbing the timing inside tight loops
(e.g. the K-step tree decode).

Each flushed PROFILE line is prefixed with `t_ns=<ns since epoch>` so that
target and draft logs from NTP/PTP-synced hosts can be aligned post-hoc to
visualize the async pipeline.

CUDA-graph safety: timing events can't be recorded into a capturing stream, so
`new_events()` returns None during capture and `emit()` no-ops on None.

Usage:
    from ssd.utils import profile

    ev = profile.new_events(3)
    if ev: ev[0].record()
    work_a()
    if ev: ev[1].record()
    work_b()
    if ev: ev[2].record()
    profile.emit("MyClass.method", ["a", "b"], ev, n_tokens=N)

    # ...at the natural boundary (end of step, end of loop, etc.):
    profile.flush()
"""

import os
import time

import torch

PROFILE = os.environ.get("SSD_PROFILE", "0") == "1"

_buffer = []  # list of (group, label, ev_start, ev_end, meta_key)


def is_active() -> bool:
    """True if profiling is on AND we're not inside a CUDA graph capture."""
    return PROFILE and not torch.cuda.is_current_stream_capturing()


def new_events(n: int):
    """Return a list of `n` timing events to record between code regions, or
    None when profiling is off / inside a CG capture. Callers guard with
    `if ev: ev[i].record()`."""
    if not is_active():
        return None
    return [torch.cuda.Event(enable_timing=True) for _ in range(n)]


class _CpuEvent:
    """CPU-only event for measuring wall-clock-blocking regions (e.g. waiting
    on a network recv). CUDA events on an idle stream do NOT capture CPU-only
    waits, so use this for those regions. API mirrors torch.cuda.Event so it
    can flow through the same emit()/flush() pipeline."""
    __slots__ = ("t",)

    def __init__(self):
        self.t = None

    def record(self):
        self.t = time.perf_counter()

    def elapsed_time(self, other: "_CpuEvent") -> float:
        return (other.t - self.t) * 1000.0


def new_cpu_events(n: int):
    """Same shape as `new_events`, but events are CPU-timed. Use for regions
    that are CPU-blocking with no GPU work (e.g. `dist.recv` on the request
    channel, or any wait_for_X)."""
    if not is_active():
        return None
    return [_CpuEvent() for _ in range(n)]


def emit(group: str, labels: list, events, **meta) -> None:
    """Queue measurements for the next flush. Each label_i corresponds to
    elapsed time from events[i] to events[i+1]. No sync.

    `meta` ends up in the print line and is part of the grouping key, so
    invocations with different metadata stay on separate lines."""
    if events is None:
        return
    if len(labels) != len(events) - 1:
        raise ValueError(
            f"profile.emit: labels={len(labels)} but events={len(events)} "
            f"(need len(labels) == len(events) - 1)"
        )
    meta_key = tuple(sorted(meta.items()))
    for i, label in enumerate(labels):
        _buffer.append((group, label, events[i], events[i + 1], meta_key))


def flush() -> None:
    """One sync, read all buffered events, print grouped by (group, meta), clear."""
    if not _buffer:
        return
    torch.cuda.synchronize()
    # Single wall-clock stamp shared by all lines from this flush. Used to align
    # target/draft logs post-hoc; assumes host clocks are NTP/PTP-synced.
    t_ns = time.time_ns()
    grouped = {}
    for group, label, ev0, ev1, meta_key in _buffer:
        grouped.setdefault((group, meta_key), []).append(
            (label, ev0.elapsed_time(ev1))
        )
    for (group, meta_key), items in grouped.items():
        total = sum(t for _, t in items)
        parts = " ".join(f"{l}={t:.3f}ms" for l, t in items)
        line = f"[PROFILE {group}] t_ns={t_ns} {parts} total={total:.3f}ms"
        if meta_key:
            line += " " + " ".join(f"{k}={v}" for k, v in meta_key)
        print(line, flush=True)
    _buffer.clear()
