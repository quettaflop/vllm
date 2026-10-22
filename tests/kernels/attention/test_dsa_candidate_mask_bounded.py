# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``apply_candidate_mask_bounded`` agrees with ``apply_candidate_mask`` over
each row's ``[start, end)`` and leaves the columns past ``end`` untouched.

The decode logits are ``max_model_len`` wide but only ``end`` columns are live;
the bounded variant skips the rest, which is sound because every top-k backend
bounds its scan by the same ends. So the contract checked here is agreement on
``[start, end)``, untouched columns beyond, and equal top-k sets.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA kernels")

BLOCK_SIZE = 8


def _inputs(rows, width, row_repeat, with_starts):
    torch.manual_seed(0)
    reqs = rows // row_repeat
    ends = torch.randint(
        width // 64, width // 32, (reqs,), device="cuda", dtype=torch.int32
    )
    ends[1] = 0  # a padding request
    starts = (
        torch.randint(0, 64, (reqs,), device="cuda", dtype=torch.int32)
        if with_starts
        else None
    )
    logits = torch.randn(rows, width, device="cuda")
    candidates = torch.randint(
        0, width // 32 // BLOCK_SIZE, (rows, 256), device="cuda", dtype=torch.int32
    )
    candidates[:, ::7] = -1
    return logits, starts, ends, candidates


@pytest.mark.parametrize("width", [8192, 1 << 20])
@pytest.mark.parametrize("row_repeat", [1, 6])
@pytest.mark.parametrize("with_starts", [False, True])
def test_bounded_mask_matches_shared_mask_in_bounds(width, row_repeat, with_starts):
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
        apply_candidate_mask_bounded,
    )

    rows = 48 * row_repeat
    logits, starts, ends, candidates = _inputs(rows, width, row_repeat, with_starts)
    ref, got = logits.clone(), logits.clone()
    apply_candidate_mask(ref, starts, ends, candidates, BLOCK_SIZE, row_repeat)
    apply_candidate_mask_bounded(got, starts, ends, candidates, BLOCK_SIZE, row_repeat)
    for r in range(rows):
        end = int(ends[r // row_repeat])
        assert torch.equal(got[r, :end], ref[r, :end]), f"row {r} differs in bounds"
        assert torch.equal(got[r, end:], logits[r, end:]), f"row {r} touched past end"


def test_bounded_mask_keeps_decode_topk():
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
        apply_candidate_mask_bounded,
    )
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda"))
    next_n, reqs, topk, width = 6, 16, 512, 1 << 20
    torch.manual_seed(0)
    ctx = torch.randint(20000, 25000, (reqs,), device="cuda", dtype=torch.int32)
    seq_lens = ctx[:, None] + torch.arange(next_n, device="cuda", dtype=torch.int32)
    vis = seq_lens.reshape(-1)
    rows = reqs * next_n
    candidates = torch.randint(
        0, 20000 // BLOCK_SIZE, (rows, 2048), device="cuda", dtype=torch.int32
    )
    logits = torch.randn(rows, width, device="cuda")
    for backend in ("persistent", "per_row"):
        outs = []
        for mask in (apply_candidate_mask, apply_candidate_mask_bounded):
            lg = logits.clone()
            mask(lg, None, vis, candidates, BLOCK_SIZE, 1)
            out = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
            get_indexer_topk(backend)(
                lg, seq_lens, next_n, out, topk, int(seq_lens.max())
            )
            outs.append(out.sort(dim=-1).values)
        assert torch.equal(outs[0], outs[1]), backend
