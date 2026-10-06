# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    _required_sm120_sparse_topk,
)
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4IndexerCache,
)
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend as DeepseekV41FlashInferMLASparseBackend,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4SparseMLABackend as DeepseekV41SparseMLABackend,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.utils import select_common_block_size


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192


@pytest.mark.parametrize(
    "capability,mla_page,indexer_page",
    [(90, [128], [64]), (100, [128], [128]), (120, [64, 128], [64, 128])],
)
def test_v41_page_sizes_preserve_other_architectures(
    monkeypatch, capability, mla_page, indexer_page
) -> None:
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda cc: cc == capability
    )
    assert (
        DeepseekV41FlashInferMLASparseBackend.get_supported_kernel_block_sizes()
        == mla_page
    )
    assert DeepseekV41IndexerBackend.get_supported_kernel_block_sizes() == indexer_page
    # The default backend and indexer must still have a common page size.
    assert (
        select_common_block_size(
            indexer_page[0], [DeepseekV41SparseMLABackend, DeepseekV41IndexerBackend]
        )
        == indexer_page[0]
    )


@pytest.mark.parametrize("compress_ratio", [1, 2])
@pytest.mark.parametrize("block_size", [64, 128])
def test_sm120_c1_c2_caches_keep_64_physical_rows(
    monkeypatch, compress_ratio, block_size
) -> None:
    """Long prompts must reach a supported DeepGEMM indexer page geometry."""
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda cc: cc == 120
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size, cache_dtype="fp8_ds_mla"),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with set_current_vllm_config(config):
        indexer = DeepseekV4IndexerCache(
            132, torch.uint8, "indexer", config.cache_config, compress_ratio
        )
        indexer_spec = indexer.get_kv_cache_spec(config)
    attn = SimpleNamespace(
        is_kv_source=True,
        kv_cache_dtype="fp8_ds_mla",
        compress_ratio=compress_ratio,
        head_dim=512,
        kv_cache_torch_dtype=torch.uint8,
        kv_page_alignment=576,
        kv_bytes_per_token=584,
    )
    attn_spec = DeepseekV4Attention.get_kv_cache_spec(attn, config)
    for spec in (attn_spec, indexer_spec):
        kernel_block = select_common_block_size(
            spec.block_size,
            [DeepseekV41FlashInferMLASparseBackend, DeepseekV41IndexerBackend],
        )
        assert spec.get_num_kernel_states(kernel_block) == 64


@pytest.mark.parametrize("num_tokens", [2, 128])
@pytest.mark.parametrize("extra_page_size", [None, 32, 64, 128])
def test_sm120_v41_page32_vision_warmup(num_tokens, extra_page_size) -> None:
    """TP8 V4.1 needs page32 and top-k 128 + 1024 vision slots."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM12x")
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

    def constant_cache(page_size, value):
        # DSv4 layout: FP8 NoPE, BF16 RoPE, then the page's UE8M0 footer.
        packed = torch.zeros((1, page_size, 1, 584), dtype=torch.uint8, device="cuda")
        flat = packed.view(-1)
        data = flat[: page_size * 576].view(page_size, 576)
        data[:, 448:] = torch.full(
            (page_size, 64), value, dtype=torch.bfloat16, device="cuda"
        ).view(torch.uint8)
        flat[page_size * 576 :] = 127  # scale = 1
        return packed

    query = torch.zeros((num_tokens, 8, 512), dtype=torch.bfloat16, device="cuda")
    indices = torch.full((num_tokens, 1152), -1, dtype=torch.int32, device="cuda")
    indices[:, :32] = torch.arange(32, device="cuda")
    kwargs = {}
    expected_rope = 1.0
    if extra_page_size is not None:
        kwargs = {
            "compressed_kv_cache": constant_cache(extra_page_size, 2),
            "extra_sparse_indices": torch.arange(16, device="cuda", dtype=torch.int32)
            .expand(num_tokens, -1)
            .contiguous(),
            "extra_sparse_topk_lens": torch.full(
                (num_tokens,), 16, dtype=torch.int32, device="cuda"
            ),
        }
        expected_rope = 4 / 3
    output = trtllm_batch_decode_sparse_mla_dsv4(
        query,
        constant_cache(32, 1),
        torch.empty(0, dtype=torch.uint8, device="cuda"),
        sparse_indices=indices,
        swa_topk_lens=torch.full((num_tokens,), 32, dtype=torch.int32, device="cuda"),
        kv_layout="NHD",
        **kwargs,
    )
    expected = torch.zeros_like(output)
    expected[..., 448:] = expected_rope
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)
