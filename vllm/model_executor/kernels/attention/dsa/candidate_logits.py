# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode indexer logits over candidate blocks only (DeepSeek-V4.1 two-level
selection).

A consumer indexer keeps the top-k of its logits among the candidate blocks
another layer selected (``candidate_blocks``: per row, block ids of
``block_size`` compressed positions, -1 for none). The default path computes
logits at every position up to the row's length and masks the rest, so its cost
grows with the context; at 1M compressed positions the paged MQA logits take
~18 ms per layer for 384 rows on sm_120. Here only the candidate positions are
scored: their index keys are gathered from the paged cache and
``sum_h w[h] * relu(q[h] . k * k_scale)`` is computed with FP8 tensor-core dots,
so the cost is fixed by the candidate count (2,048 x 8 per row) instead.

The index-key cache is the FP8 paged layout DeepGEMM reads: each page holds
``page_rows`` rows of ``head_dim`` FP8 values, then ``page_rows`` fp32 scales.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _candidate_logits_kernel(
    q,
    stride_q_row,
    stride_q_head,
    weights,
    stride_w_row,
    kv,
    stride_kv_page,
    block_table,
    stride_bt_row,
    candidates,
    stride_cand_row,
    lens,
    out,
    stride_out_row,
    NEXT_N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    CAND_BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = row // NEXT_N
    vis = tl.load(lens + row)
    slots = tl.program_id(1) * TILE + tl.arange(0, TILE)
    block = tl.load(candidates + row * stride_cand_row + slots // CAND_BLOCK)
    pos = block.to(tl.int64) * CAND_BLOCK + slots % CAND_BLOCK
    valid = (block >= 0) & (pos < vis)
    page = tl.load(
        block_table + req * stride_bt_row + pos // PAGE_ROWS, mask=valid, other=0
    ).to(tl.int64)
    page_row = pos % PAGE_ROWS
    base = kv + page * stride_kv_page
    d = tl.arange(0, D)
    k = tl.load(
        base[:, None] + page_row[:, None] * D + d[None, :],
        mask=valid[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    k_scale = tl.load(
        (base + PAGE_ROWS * D + page_row * 4).to(tl.pointer_type(tl.float32)),
        mask=valid,
        other=0.0,
    )
    h = tl.arange(0, H)
    q_tile = tl.load(q + row * stride_q_row + h[:, None] * stride_q_head + d[None, :])
    scores = tl.dot(k, tl.trans(q_tile))  # [TILE, H]
    scores = tl.maximum(scores * k_scale[:, None], 0.0)
    w = tl.load(weights + row * stride_w_row + h)
    logits = tl.sum(scores * w[None, :], axis=1)
    tl.store(
        out + row * stride_out_row + slots,
        tl.where(valid, logits, -float("inf")),
    )


def candidate_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
) -> torch.Tensor:
    """Logits at each row's candidate positions.

    Args:
        q: ``[batch, next_n, heads, head_dim]`` FP8 E4M3 index queries (the
            per-token scale folded into ``weights``).
        kv_cache: ``[num_pages, page_rows, 1, head_dim + 4]`` uint8 FP8 index
            keys in DeepGEMM's paged layout.
        weights: ``[batch * next_n, heads]`` fp32 head weights.
        seq_lens: ``[batch, next_n]`` int32 visible lengths per row.
        block_table: ``[batch, max_pages]`` int32 pages of each request.
        candidate_blocks: ``[batch * next_n, num_candidates]`` int32 request-local
            block ids of ``candidate_block_size`` positions, -1 for none.

    Returns:
        ``[batch * next_n, num_candidates * candidate_block_size]`` fp32 logits,
        -inf where the candidate is absent or past the row's length. Column
        ``j`` is position ``candidate_blocks[row, j // block] * block + j % block``.
    """
    batch, next_n, heads, head_dim = q.shape
    rows = batch * next_n
    page_rows = kv_cache.shape[1]
    width = candidate_blocks.shape[1] * candidate_block_size
    assert q.dtype == torch.float8_e4m3fn and kv_cache.dtype == torch.uint8
    assert kv_cache.shape[-1] == head_dim + 4
    assert weights.shape == (rows, heads) and weights.dtype == torch.float32
    assert candidate_blocks.shape[0] == rows
    assert seq_lens.shape == (batch, next_n) and seq_lens.is_contiguous()
    tile = 128
    assert width % tile == 0 and tile % candidate_block_size == 0
    out = torch.empty((rows, width), dtype=torch.float32, device=q.device)
    if rows == 0:
        return out
    q_rows = q.reshape(rows, heads, head_dim)
    _candidate_logits_kernel[(rows, width // tile)](
        q_rows,
        q_rows.stride(0),
        q_rows.stride(1),
        weights,
        weights.stride(0),
        kv_cache,
        kv_cache.stride(0),
        block_table,
        block_table.stride(0),
        candidate_blocks,
        candidate_blocks.stride(0),
        seq_lens,
        out,
        out.stride(0),
        NEXT_N=next_n,
        H=heads,
        D=head_dim,
        PAGE_ROWS=page_rows,
        CAND_BLOCK=candidate_block_size,
        TILE=tile,
        num_warps=4,
    )
    return out


@triton.jit
def _candidate_topk_to_positions_kernel(
    cols,
    stride_cols_row,
    logits,
    stride_logits_row,
    candidates,
    stride_cand_row,
    out,
    stride_out_row,
    K: tl.constexpr,
    K_P2: tl.constexpr,
    CAND_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, K_P2)
    in_k = i < K
    col = tl.load(cols + row * stride_cols_row + i, mask=in_k, other=-1)
    ok = in_k & (col >= 0)
    logit = tl.load(
        logits + row * stride_logits_row + col, mask=ok, other=-float("inf")
    )
    block = tl.load(
        candidates + row * stride_cand_row + col // CAND_BLOCK, mask=ok, other=-1
    )
    ok = ok & (logit > -float("inf")) & (block >= 0)
    pos = block * CAND_BLOCK + col % CAND_BLOCK
    # Attention reads a row's first `count` entries, so valid ones go first.
    dst = tl.cumsum(ok.to(tl.int32), axis=0) - 1
    tl.store(out + row * stride_out_row + dst, pos, mask=ok)
    count = tl.sum(ok.to(tl.int32), axis=0)
    tl.store(out + row * stride_out_row + i, -1, mask=in_k & (i >= count))


def candidate_topk_to_positions(
    topk_cols: torch.Tensor,
    logits: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    out: torch.Tensor,
) -> None:
    """Map top-k columns of ``candidate_mqa_logits`` back to request-local
    positions in ``out``, valid ones first and -1 after (columns whose logit
    is -inf are dropped)."""
    rows, k = topk_cols.shape
    if rows == 0:
        return
    _candidate_topk_to_positions_kernel[(rows,)](
        topk_cols,
        topk_cols.stride(0),
        logits,
        logits.stride(0),
        candidate_blocks,
        candidate_blocks.stride(0),
        out,
        out.stride(0),
        K=k,
        K_P2=triton.next_power_of_2(k),
        CAND_BLOCK=candidate_block_size,
    )
