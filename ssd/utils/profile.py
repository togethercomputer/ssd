"""Deferred CUDA-event profiling, gated by `SSD_PROFILE=1`.

Records events without per-call `torch.cuda.synchronize()` — sync happens once
at `flush()`. This avoids GPU drains perturbing the timing inside tight loops
(e.g. the K-step tree decode).

Each flushed PROFILE line is prefixed with `t_ns=<ns since epoch>` so that
target and draft logs from NTP/PTP-synced hosts can be aligned post-hoc to
visualize the async pipeline.

CUDA-graph safety: timing events can't be recorded into a capturing stream, so
`new_events()` returns None during capture and `emit()` no-ops on None.

----------------------------------------------------------------------------
Chrome trace output
----------------------------------------------------------------------------

When SSD_PROFILE_TRACE=1, every flushed segment is also buffered as a
chrome://tracing "X" event (complete event with duration). Load the dumped
JSON in chrome://tracing or ui.perfetto.dev to see the async pipeline
visually — two process rows (target / draft), sub-rows per group
(sglang_draft, sglang_decode, draft.spec_iter, draft.idle, …), bars showing
actual segment timings that line up across hosts.

  SSD_PROFILE_TRACE=1        # enable chrome trace dump
  SSD_PROFILE_TRACE_NAME=X   # friendly name for this process in the trace
                             # (e.g. "target_rank0" or "draft")
  SSD_PROFILE_TRACE_OUT=path # destination JSON (default
                             # /tmp/ssd_trace_<name>_<pid>.json)

To merge target + draft traces into one file viewable side-by-side:
  jq -s '{traceEvents: map(.traceEvents) | add}' \\
      target_trace.json draft_trace.json > merged.json

----------------------------------------------------------------------------
When does the trace get written? Three triggers, in increasing robustness
----------------------------------------------------------------------------

1. **Clean exit** — `atexit` hook fires on normal return, `sys.exit()`, or
   Ctrl-C that unwinds to top of interpreter. The most common case.

2. **Ctrl-C (SIGINT) is mostly fine but has known failure modes:**
   - Process stuck in a NCCL collective: the C call holds the GIL, so the
     KeyboardInterrupt only delivers when the call returns. If the peer is
     gone, the call never returns. A second Ctrl-C triggers a harder abort
     that may skip atexit.
   - Multi-process workers (sglang serve spawns TP workers): SIGINT only
     reaches the worker processes if your shell delivers to the process
     group. If you backgrounded the server or run it under a wrapper, the
     children may not get SIGINT cleanly. Rank-0 is usually the one with
     events, so this often works out.
   - `kill -9` (SIGKILL) and unhandled crashes skip atexit entirely.

3. **SIGUSR1 on-demand snapshot (most robust).** Send `kill -USR1 <pid>` to
   the running process and the trace is written to disk immediately. The
   process keeps running. Repeat as often as you want — each dump overwrites
   the previous file with everything-so-far. The handler is registered when
   SSD_PROFILE_TRACE=1; on startup the module prints the PID and target
   path so you know what to signal. Useful when:
   - You want a snapshot of a long-running benchmark mid-flight.
   - You're about to `kill -9` and want to grab a trace first.
   - sglang's clean shutdown is unreliable in your setup.

   Workflow:
       SSD_PROFILE=1 SSD_PROFILE_TRACE=1 ... sglang serve ... &
       # log will print: [profile] pid=12345 ... — send SIGUSR1 to snapshot
       kill -USR1 12345        # writes target_trace.json
       # ... benchmark continues ...
       kill -USR1 12345        # overwrites with newer snapshot
       # eventually:
       kill -INT 12345         # clean shutdown; atexit also dumps

   Caveat: signal.signal() registers on the main thread only. If something
   else in the process already installed a SIGUSR1 handler, the last one
   wins. SIGUSR1 is conventionally unused, so this is unlikely.

----------------------------------------------------------------------------
Usage
----------------------------------------------------------------------------
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

import atexit
import json
import os
import time

import torch

PROFILE = os.environ.get("SSD_PROFILE", "0") == "1"
TRACE = os.environ.get("SSD_PROFILE_TRACE", "0") == "1"
_TRACE_NAME = os.environ.get("SSD_PROFILE_TRACE_NAME", f"proc_{os.getpid()}")
_TRACE_OUT = os.environ.get(
    "SSD_PROFILE_TRACE_OUT", f"/tmp/ssd_trace_{_TRACE_NAME}_{os.getpid()}.json"
)
_TRACE_PID = os.getpid()

_buffer = []  # list of (group, label, ev_start, ev_end, meta_key)
_trace_events = []  # list of chrome://tracing event dicts


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

        if TRACE:
            # Anchor this group's last segment end at t_ns, then walk back so
            # all segments share their measured durations. Cross-group skew
            # within a single flush is bounded by the inter-group gap (μs),
            # acceptable for chrome://tracing visualization.
            meta_args = dict(meta_key)
            seg_end_us = t_ns / 1000.0  # chrome wants microseconds
            for label, dur_ms in reversed(items):
                dur_us = dur_ms * 1000.0
                seg_start_us = seg_end_us - dur_us
                _trace_events.append({
                    "name": label,
                    "cat": group,
                    "ph": "X",
                    "ts": seg_start_us,
                    "dur": dur_us,
                    "pid": _TRACE_PID,
                    "tid": group,
                    "args": meta_args,
                })
                seg_end_us = seg_start_us

    _buffer.clear()


def dump_trace(path: str | None = None) -> None:
    """Write the accumulated chrome://tracing events to `path` (defaults to
    SSD_PROFILE_TRACE_OUT). No-op if SSD_PROFILE_TRACE is off or no events."""
    if not TRACE or not _trace_events:
        return
    out = path or _TRACE_OUT
    # Emit one process_name metadata event so chrome://tracing labels the
    # process row instead of showing the raw pid.
    events = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": _TRACE_PID,
            "tid": 0,
            "args": {"name": _TRACE_NAME},
        }
    ] + _trace_events
    with open(out, "w") as f:
        json.dump({"traceEvents": events}, f)
    print(
        f"[profile] Wrote chrome trace: {out} ({len(_trace_events)} events)",
        flush=True,
    )


if TRACE:
    # Trigger 1: normal exit (sys.exit, return-from-main, KeyboardInterrupt
    # that unwinds the interpreter).
    atexit.register(dump_trace)

    # Trigger 2: on-demand snapshot via `kill -USR1 <pid>`. The process keeps
    # running; each signal overwrites _TRACE_OUT with everything-so-far. See
    # the module docstring for caveats (main-thread requirement, GIL during
    # NCCL calls). Wrapped in try/except so import doesn't fail in unusual
    # environments (e.g. non-main thread import).
    import signal
    try:
        signal.signal(signal.SIGUSR1, lambda *_: dump_trace())
        _signal_help = " — send SIGUSR1 to snapshot trace"
    except (ValueError, OSError) as e:
        # ValueError: not main thread. OSError: signal unsupported on this OS.
        _signal_help = f" — SIGUSR1 handler not installed ({e})"

    print(
        f"[profile] pid={_TRACE_PID} name={_TRACE_NAME} "
        f"out={_TRACE_OUT}{_signal_help}",
        flush=True,
    )
