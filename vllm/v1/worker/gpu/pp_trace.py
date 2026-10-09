# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in PP timing without synchronizing the steady-state CUDA streams."""

import atexit
import hashlib
import json
import os
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import torch

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.v1.core.sched.output import SchedulerOutput


class PPTrace:
    def __init__(self) -> None:
        self.step = 0
        self.pending: deque[tuple[dict, dict[str, torch.cuda.Event]]] = deque()
        self.current: tuple[dict, dict[str, torch.cuda.Event]] | None = None
        self.anchor: torch.cuda.Event | None = None
        self.anchor_ns = 0
        self._files = ExitStack()
        atexit.register(self._files.close)
        self.output = self._files.enter_context(
            Path(f"/tmp/vllm-pp-trace-{os.getpid()}.jsonl").open("w")  # noqa: SIM115
        )

    def begin(self, output: SchedulerOutput) -> None:
        if self.anchor is None:
            self.anchor = torch.cuda.Event(enable_timing=True)
            torch.cuda.current_stream().synchronize()
            lo = time.monotonic_ns()
            self.anchor.record()
            self.anchor.synchronize()
            hi = time.monotonic_ns()
            self.anchor_ns = (lo + hi) // 2
            self.output.write(
                json.dumps(
                    {
                        "anchor_lo_ns": lo,
                        "anchor_hi_ns": hi,
                        "pp_rank": get_pp_group().rank_in_group,
                        "tp_rank": get_tp_group().rank_in_group,
                    }
                )
                + "\n"
            )
        if self.current is not None:
            self.pending.append(self.current)
        self.flush()
        self.step += 1
        row = {
            "step": self.step,
            "num_reqs": len(output.num_scheduled_tokens),
            "num_tokens": output.total_num_scheduled_tokens,
            "requests": [
                hashlib.blake2b(r.encode(), digest_size=8).hexdigest()
                for r in output.num_scheduled_tokens
            ],
        }
        self.current = row, {}
        self.mark("execute_start")

    def mark(self, name: str, stream: torch.cuda.Stream | None = None) -> None:
        assert self.current is not None
        row, events = self.current
        row[name + "_host_ns"] = time.monotonic_ns()
        event = torch.cuda.Event(enable_timing=True)
        event.record(stream)
        events[name] = event

    def flush(self) -> None:
        assert self.anchor is not None
        wrote = False
        while self.pending:
            row, events = self.pending[0]
            if not all(event.query() for event in events.values()):
                break
            self.pending.popleft()
            for name, event in events.items():
                row[name + "_gpu_ns"] = self.anchor_ns + round(
                    self.anchor.elapsed_time(event) * 1_000_000
                )
            self.output.write(json.dumps(row) + "\n")
            wrote = True
        if wrote:
            self.output.flush()
