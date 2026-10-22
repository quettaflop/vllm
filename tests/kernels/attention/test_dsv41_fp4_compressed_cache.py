# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""rope_quant_insert's V41_FP4 path (288 B compressed rows, SM120) writes the
rows FlashInfer's own full-page FP4 pack writes for the same RoPE'd latent, at
both compress ratios, through a padded page stride, skipping -1 slots and the
non-boundary tokens of a ratio-2 group."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="V41_FP4 compressed cache is SM120-only",
)

ROWS, PAGES, PAD = 64, 8, 512


@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_fp4_insert_matches_flashinfer_pack(compress_ratio):
    flashinfer_mla = pytest.importorskip("flashinfer.mla")
    if not hasattr(flashinfer_mla, "dsv41_fp4_quantize_pack_sparse_mla_cache"):
        pytest.skip("FlashInfer without the V41_FP4 cache ops")
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
        _rope_plain_insert_kernel,
        rope_quant_insert,
    )

    torch.manual_seed(0)
    n = 700
    latent = torch.randn(n, 512, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(3, 3 + n, device="cuda", dtype=torch.int64)
    cos_sin = torch.randn(4096, 64, dtype=torch.float32, device="cuda")
    group = (positions - 3) // compress_ratio
    slot = group.clone()
    slot[group % 17 == 5] = -1
    slot = torch.where(slot < PAGES * ROWS, slot, -1)
    page_bytes = ROWS * 288
    stride = page_bytes + PAD
    storage = torch.full((PAGES * stride,), 0xAB, dtype=torch.uint8, device="cuda")
    cache = storage.as_strided((PAGES, ROWS, 288), (stride, 288, 1))
    rope_quant_insert(latent, positions, cos_sin, cache, slot, compress_ratio)

    ref_rows = torch.zeros(PAGES, ROWS, 512, dtype=torch.bfloat16, device="cuda")
    _rope_plain_insert_kernel[(n,)](
        latent,
        positions,
        cos_sin,
        ref_rows,
        slot,
        None,
        COS_STRIDE=cos_sin.stride(0),
        CACHE_STRIDE=ref_rows.stride(0),
        ROW_STRIDE=ref_rows.stride(1),
        CACHE_BLOCK=ROWS,
        COMPRESS_RATIO=compress_ratio,
        STORE_FP8=False,
        num_warps=4,
    )
    ref = flashinfer_mla.dsv41_fp4_quantize_pack_sparse_mla_cache(
        ref_rows, kv_layout="NHD"
    ).view(PAGES, -1)
    got = storage.view(PAGES, stride)
    publish = (slot >= 0) & ((positions + 1) % compress_ratio == 0)
    written = torch.zeros(PAGES * ROWS, dtype=torch.bool, device="cuda")
    written[slot[publish]] = True
    written = written.view(PAGES, ROWS)
    assert int(written.sum()) > 0
    for p in range(PAGES):
        for r in range(ROWS):
            if not written[p, r]:
                continue
            d, s = r * 256, ROWS * 256 + r * 32
            assert torch.equal(got[p, d : d + 256], ref[p, d : d + 256]), (p, r)
            assert torch.equal(got[p, s : s + 32], ref[p, s : s + 32]), (p, r)
    assert bool((got[:, page_bytes:] == 0xAB).all()), "page padding was written"
