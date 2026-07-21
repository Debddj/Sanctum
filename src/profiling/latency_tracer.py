"""Per-stage latency tracer, exported as spans.

Instruments each pipeline stage (capture → landmarks → classifier → dispatch → render)
with high-resolution timers. Results are collected for benchmarking reports.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class Span:
    """A single timing span for a pipeline stage."""

    name: str
    start_ns: int = 0
    end_ns: int = 0

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


@dataclass
class FrameTrace:
    """Collection of spans for a single frame's pipeline execution."""

    frame_id: int
    spans: list[Span] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans)


class LatencyTracer:
    """Collects per-stage timing spans across frames."""

    def __init__(self) -> None:
        self._traces: list[FrameTrace] = []
        self._current_frame: int = 0

    @contextmanager
    def trace(self, stage_name: str) -> Generator[Span, None, None]:
        """Context manager that times a pipeline stage.

        Usage:
            with tracer.trace("landmarks") as span:
                landmarks = extractor.extract(frame)
        """
        span = Span(name=stage_name, start_ns=time.perf_counter_ns())
        yield span
        span.end_ns = time.perf_counter_ns()

        # Append to current frame trace
        if not self._traces or self._traces[-1].frame_id != self._current_frame:
            self._traces.append(FrameTrace(frame_id=self._current_frame))
        self._traces[-1].spans.append(span)

    def next_frame(self) -> None:
        """Advance to the next frame."""
        self._current_frame += 1

    def get_traces(self) -> list[FrameTrace]:
        """Return all collected traces."""
        return self._traces.copy()

    def clear(self) -> None:
        """Reset all traces."""
        self._traces.clear()
        self._current_frame = 0
