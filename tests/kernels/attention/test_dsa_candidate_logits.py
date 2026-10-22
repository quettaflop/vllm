# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-only decode indexer logits (``candidate_mqa_logits``) against a
torch reference, and the candidate top-k against the full-width path
(paged MQA logits over the whole context, candidate mask, top-k)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")

HEADS, DIM, PAGE_ROWS, CAND_BLOCK, NUM_CAND, TOPK = 32, 128, 64, 8, 2048, 512


def _paged_fp8_cache(keys: torch.Tensor) -> torch.Tensor:
    """[pages, rows, dim] bf16 -> DeepGEMM's paged FP8 index cache (values, then
    one fp32 scale per row), viewed as [pages, rows, 1, dim + 4] uint8."""
    pages, rows, dim = keys.shape
    scale = keys.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4) / 448.0
    values = (keys / scale).to(torch.float8_e4m3fn)
    cache = torch.empty(pages, rows * (dim + 4), dtype=torch.uint8, device=keys.device)
    cache[:, : rows * dim] = values.view(pages, rows * dim).view(torch.uint8)
    cache[:, rows * dim :] = scale.view(pages, rows).view(torch.uint8)
    return cache.view(pages, rows, 1, dim + 4)


def _inputs(batch, next_n, ctx):
    torch.manual_seed(0)
    pages_per_req = (ctx + PAGE_ROWS - 1) // PAGE_ROWS
    keys = torch.randn(batch * pages_per_req, PAGE_ROWS, DIM, device="cuda") / 4
    cache = _paged_fp8_cache(keys.bfloat16())
    block_table = (
        torch.randperm(batch * pages_per_req, device="cuda")
        .to(torch.int32)
        .view(batch, pages_per_req)
    )
    q = (torch.randn(batch, next_n, HEADS, DIM, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(batch * next_n, HEADS, device="cuda")
    lens = (
        torch.full((batch, 1), ctx, device="cuda", dtype=torch.int32)
        - next_n
        + 1
        + torch.arange(next_n, device="cuda", dtype=torch.int32)
    ).contiguous()
    blocks = (ctx + CAND_BLOCK - 1) // CAND_BLOCK
    cand = torch.stack(
        [
            torch.randperm(blocks, device="cuda")[:NUM_CAND]
            for _ in range(batch * next_n)
        ]
    ).to(torch.int32)
    if blocks < NUM_CAND:
        cand = torch.cat(
            [
                cand,
                torch.full(
                    (batch * next_n, NUM_CAND - blocks),
                    -1,
                    device="cuda",
                    dtype=torch.int32,
                ),
            ],
            dim=1,
        )
    cand[:, ::97] = -1
    return q, cache, block_table, weights, lens, cand


def _reference(q, cache, block_table, weights, lens, cand):
    batch, next_n = q.shape[:2]
    pages, rows = cache.shape[:2]
    flat = cache.view(pages, -1)
    values = flat[:, : rows * DIM].view(torch.float8_e4m3fn).view(pages, rows, DIM)
    scales = flat[:, rows * DIM :].contiguous().view(torch.float32).view(pages, rows)
    out = torch.full(
        (batch * next_n, cand.shape[1] * CAND_BLOCK), float("-inf"), device="cuda"
    )
    for r in range(batch * next_n):
        b = r // next_n
        cols = torch.arange(out.shape[1], device="cuda")
        block = cand[r, cols // CAND_BLOCK].long()
        pos = block * CAND_BLOCK + cols % CAND_BLOCK
        ok = (block >= 0) & (pos < lens.view(-1)[r])
        pos = pos[ok]
        page = block_table[b, pos // PAGE_ROWS].long()
        k = values[page, pos % PAGE_ROWS].float() * scales[page, pos % PAGE_ROWS, None]
        s = (q[b, r % next_n].float() @ k.T).relu()  # [heads, n]
        out[r, ok] = (weights[r, :, None] * s).sum(0)
    return out


@pytest.mark.parametrize("ctx", [3000, 40000])
def test_candidate_logits_match_reference(ctx):
    from vllm.model_executor.kernels.attention.dsa.candidate_logits import (
        candidate_mqa_logits,
    )

    q, cache, block_table, weights, lens, cand = _inputs(2, 3, ctx)
    got = candidate_mqa_logits(q, cache, weights, lens, block_table, cand, CAND_BLOCK)
    ref = _reference(q, cache, block_table, weights, lens, cand)
    assert torch.equal(torch.isinf(got), torch.isinf(ref))
    finite = torch.isfinite(ref)
    torch.testing.assert_close(got[finite], ref[finite], rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
def test_candidate_topk_matches_full_width_path():
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask_bounded,
    )
    from vllm.model_executor.kernels.attention.dsa.candidate_logits import (
        candidate_mqa_logits,
        candidate_topk_to_positions,
    )
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_num_sms,
        get_paged_mqa_logits_metadata,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda"))
    batch, next_n, ctx = 4, 6, 40000
    q, cache, block_table, weights, lens, cand = _inputs(batch, next_n, ctx)
    rows = batch * next_n
    topk = get_indexer_topk("per_row")

    sched = get_paged_mqa_logits_metadata(lens, PAGE_ROWS, get_num_sms())
    full = fp8_fp4_paged_mqa_logits(
        (q, None),
        cache,
        weights,
        lens,
        block_table,
        sched,
        1 << 16,
        clean_logits=False,
    )
    apply_candidate_mask_bounded(full, None, lens.view(-1), cand, CAND_BLOCK, 1)
    full_idx = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    topk(full, lens, next_n, full_idx, TOPK, int(lens.max()))

    cand_logits = candidate_mqa_logits(
        q, cache, weights, lens, block_table, cand, CAND_BLOCK
    )
    cols = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    topk(
        cand_logits,
        torch.full_like(lens, cand_logits.shape[1]),
        next_n,
        cols,
        TOPK,
        cand_logits.shape[1],
    )
    cand_idx = torch.empty(rows, TOPK, device="cuda", dtype=torch.int32)
    candidate_topk_to_positions(cols, cand_logits, cand, CAND_BLOCK, cand_idx)

    for r in range(rows):
        a = set(full_idx[r].tolist()) - {-1}
        b = set(cand_idx[r].tolist()) - {-1}
        # Two FP8 dot orders: allow a few near-tie swaps at the top-k boundary.
        assert len(a & b) >= 0.98 * TOPK, (r, len(a & b))
