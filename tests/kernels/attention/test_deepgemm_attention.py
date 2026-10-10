# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    _ceil_to_ue8m0,
    calc_diff,
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    native_next_n_supported,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    # x: (num_blocks, block_size, 1, head_dim)
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        dtype=torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: tuple, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def _generate_cp_test_data(seq_len: int, seq_len_kv: int):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    ke = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def _ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    seq_len_kv = kv.shape[0]

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
@pytest.mark.parametrize("clean_logits", [True, False])
def test_deepgemm_fp8_mqa_logits(clean_logits: bool):
    torch.manual_seed(0)
    random.seed(0)
    num_heads, head_dim = 32, 128
    for seq_len in (512,):
        for seq_len_kv in (1024,):
            for disable_cp in (False, True):
                q = torch.randn(
                    seq_len,
                    num_heads,
                    head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                kv = torch.randn(
                    seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16
                )
                weights = torch.randn(
                    seq_len, num_heads, device="cuda", dtype=torch.float32
                )

                if disable_cp:
                    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
                    ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (
                        seq_len_kv - seq_len
                    )
                else:
                    ks, ke = _generate_cp_test_data(seq_len, seq_len_kv)

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)
                logits = fp8_fp4_mqa_logits(
                    (q_fp8, None), kv_fp8, weights, ks, ke, clean_logits=clean_logits
                )

                ref_logits = _ref_fp8_mqa_logits(
                    q=q,
                    kv=kv,
                    weights=weights,
                    cu_seqlen_ks=ks,
                    cu_seqlen_ke=ke,
                )
                ref_neginf_mask = ref_logits == float("-inf")

                if clean_logits:
                    neginf_mask = logits == float("-inf")
                    assert torch.equal(neginf_mask, ref_neginf_mask)

                ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                logits = logits.masked_fill(ref_neginf_mask, 0)
                diff = calc_diff(logits, ref_logits)
                assert diff < 1e-3, f"{diff=}"


def _ref_fp8_fp4_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, _, _ = q.size()
    _, block_size, _, _ = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device="cuda",
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("select_k", [128, 512, 2048])
@pytest.mark.parametrize("candidate_mode", ["none", "write", "read"])
def test_chunked_decode_topk_matches_full_batch_and_graph_replay(
    monkeypatch, workspace_init, next_n, select_k, candidate_mode
):
    if not native_next_n_supported(next_n):
        pytest.skip(f"next_n={next_n} has no native kernel on this architecture")

    from vllm.config import CUDAGraphMode
    from vllm.model_executor.layers import sparse_attn_indexer as indexer
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    torch.manual_seed(17)
    batch_size, heads, dim, block_size, max_len = 9, 32, 128, 64, 65536
    q = torch.randn(batch_size, next_n, heads, dim, device="cuda").to(
        torch.float8_e4m3fn
    )
    cache = kv_cache_cast_to_fp8(torch.randn(96, block_size, 1, dim, device="cuda"))
    weights = torch.rand(batch_size * next_n, heads, device="cuda")
    seq_lens = torch.tensor(
        [500, 3000, 4096, 2049, 1024, 4000, 2500, 3200, 0],
        dtype=torch.int32,
        device="cuda",
    )[:, None].repeat(1, next_n)
    if next_n > 1:
        seq_lens[:, 0] = (seq_lens[:, 0] - 1).clamp_min(0)
    original_seq_lens = seq_lens.clone()
    block_table = torch.arange(64, dtype=torch.int32, device="cuda").repeat(
        batch_size, 1
    )
    metadata = get_paged_mqa_logits_metadata(seq_lens, block_size, get_num_sms())
    expected = torch.empty(
        batch_size * next_n, select_k, dtype=torch.int32, device="cuda"
    )
    actual = torch.empty_like(expected)
    candidates = (
        torch.tensor([0, 2, 3, -1], dtype=torch.int32, device="cuda").repeat(
            batch_size * next_n, 1
        )
        if candidate_mode != "none"
        else None
    )
    if candidates is not None:
        candidates[1::2, 1] = 1
    decode = DeepSeekV32IndexerDecodeMetadata(
        block_table=block_table,
        seq_lens=seq_lens,
        decode_lens=torch.full((batch_size,), next_n, dtype=torch.int32, device="cuda"),
        requires_padding=False,
        schedule_metadata=metadata,
    )
    attn_metadata = DeepseekV32IndexerMetadata(
        seq_lens=seq_lens,
        max_seq_len=4096,
        slot_mapping=torch.zeros(batch_size * next_n, dtype=torch.int64, device="cuda"),
        num_decodes=batch_size,
        num_decode_tokens=batch_size * next_n,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )
    monkeypatch.setattr(
        indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"test_indexer": attn_metadata},
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        ),
    )
    hidden = torch.empty(batch_size * next_n, 1, device="cuda")
    allocations = []
    paged_logits = indexer.fp8_fp4_paged_mqa_logits

    def record_logits(*args, **kwargs):
        logits = paged_logits(*args, **kwargs)
        allocations.append(logits.numel() * logits.element_size())
        return logits

    monkeypatch.setattr(indexer, "fp8_fp4_paged_mqa_logits", record_logits)

    def run(dst):
        indexer.sparse_attn_indexer(
            hidden,
            "test_indexer",
            cache.squeeze(-2),
            q.reshape(batch_size * next_n, heads, dim),
            None,
            None,
            weights,
            dim,
            "ue8m0",
            select_k,
            dim,
            max_len,
            0,
            dst,
            True,
            False,
            "",
            candidate_blocks=candidates,
            candidate_block_size=1024 if candidates is not None else 0,
            candidate_write=candidate_mode == "write",
        )

    def selected_indices(dst):
        if candidate_mode == "read":
            assert candidates is not None
            # With fewer finite candidates than k, top-k may choose different
            # tied -inf entries. Compare every unmasked selection exactly.
            blocks = dst.clamp_min(0) // 1024
            allowed = (blocks[..., None] == candidates[:, None, :]).any(-1)
            dst = torch.where(allowed, dst, -1)
        return dst.sort(-1).values

    monkeypatch.setattr(indexer.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)
    run(expected)
    # 1 MiB forces several chunks, including an incomplete final chunk.
    monkeypatch.setattr(indexer.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
    allocations.clear()
    run(actual)
    assert max(allocations) <= 1024**2
    torch.testing.assert_close(selected_indices(actual), selected_indices(expected))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(actual)
    for length in (2048, 4096):
        seq_lens.copy_(original_seq_lens.clamp_max(length))
        weights.copy_(torch.rand_like(weights))
        metadata.copy_(
            get_paged_mqa_logits_metadata(seq_lens, block_size, get_num_sms())
        )
        monkeypatch.setattr(indexer.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)
        run(expected)
        graph.replay()
        torch.testing.assert_close(selected_indices(actual), selected_indices(expected))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
# next_n = 1 + num_speculative_tokens, so next_n=4 is MTP=3 (issue #35878).
@pytest.mark.parametrize("batch_size,next_n", [(4, 1), (2, 2), (2, 4)])
def test_deepgemm_fp8_fp4_paged_mqa_logits(batch_size: int, next_n: int):
    if not native_next_n_supported(next_n):
        pytest.skip(f"next_n={next_n} has no native kernel on this architecture")

    # NOTE: clean_logits=True is incompatible with the 2D context_lens
    # required by csrc/apis/attention.hpp; only the False path is exercised.
    clean_logits = False
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    for heads, index_dim in [(32, 128)]:
        for avg_kv in (2048,):
            num_blocks, blocksize = max_model_len * 2, 64

            q = torch.randn(
                (batch_size, next_n, heads, index_dim),
                device="cuda",
                dtype=torch.bfloat16,
            )
            kv_cache = torch.randn(
                (num_blocks, blocksize, 1, index_dim),
                device="cuda",
                dtype=torch.bfloat16,
            )
            weights = torch.randn(
                (batch_size * next_n, heads),
                device="cuda",
                dtype=torch.float32,
            )

            context_lens = (
                torch.randint(int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,))
                .cuda()
                .to(torch.int32)
            )
            max_block_len = (
                (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
            )
            block_tables = torch.zeros(
                (batch_size, max_block_len),
                device="cuda",
                dtype=torch.int32,
            )

            counter = 0
            block_idx_pool = list(range(num_blocks))
            random.shuffle(block_idx_pool)
            for i in range(batch_size):
                ctx_len = int(context_lens[i].item())
                for j in range((ctx_len + blocksize - 1) // blocksize):
                    block_tables[i][j] = block_idx_pool[counter]
                    counter += 1

            q_fp8 = q.to(torch.float8_e4m3fn)
            kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

            # deep_gemm paged MQA logits requires 2D context_lens of
            # shape (B, next_n) (csrc/apis/attention.hpp:332-335);
            # see indexer.py:607-608. For each batch/next_n token, the
            # effective context length is context_lens[b] - next_n + j + 1.
            next_n_arange = torch.arange(next_n, device="cuda", dtype=torch.int32)
            context_lens_2d = (
                context_lens.unsqueeze(-1) - next_n + 1 + next_n_arange
            ).contiguous()
            schedule_metadata = get_paged_mqa_logits_metadata(
                context_lens_2d,
                blocksize,
                get_num_sms(),
            )
            logits = fp8_fp4_paged_mqa_logits(
                (q_fp8, None),
                kv_cache_fp8,
                weights,
                context_lens_2d,
                block_tables,
                schedule_metadata,
                max_model_len,
                clean_logits=clean_logits,
            )

            ref_logits = _ref_fp8_fp4_paged_mqa_logits(
                q,
                kv_cache,
                weights,
                context_lens,
                block_tables,
                max_model_len,
            )

            positions = (
                torch.arange(max_model_len, device="cuda")
                .unsqueeze(0)
                .expand(batch_size * next_n, -1)
            )
            row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
            next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
            mask = positions <= (
                context_lens[row_indices] - next_n + next_n_offset
            ).unsqueeze(1)

            logits = logits.masked_fill(~mask, 0)
            ref_logits = ref_logits.masked_fill(~mask, 0)
            diff = calc_diff(logits, ref_logits)
            assert diff < 1e-3, f"{diff=}"
