# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline splits inside DeepSeek V4.1 KV-sharing groups.

Only the KV source layers (``kv_source_layer_ids``, e.g. 2/8/14/20) own a
compressed-KV cache and an indexer K cache; every later compressed layer up to
the next source reads the source's caches, and layers that are not index
sources read the top-k rows the latest index source wrote. A pipeline stage
that starts inside such a group therefore needs state another stage produces.

This module lets any split work by relaying that state down the pipeline:

- A stage keeps a *mirror* of each upstream source cache its layers read. The
  mirror registers under the source's layer names with an identical KV-cache
  spec, so the KV-cache manager gives it the same block ids, and the
  forward-context lookups the consumer layers already do find it.
- The source stage exports the step's compressed latent (bf16, before RoPE and
  quantization) and, for index-K mirrors, the indexer's ``wk`` projection. The
  receiving stage writes them into its mirror with the same insert kernels and
  its own slot mappings, so the mirror bytes equal the source's.
- The shared top-k and candidate-block rows travel as they are.

Middle stages relay whatever later stages need. Decoder replay turns itself off
when its cut layer is mirrored (see ``DeepseekV4Model._decoder_replay_supported``).
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v41.attention import (
    DeepseekV4Attention,
    DeepseekV4IndexerCache,
    _indexer_k_cache_head_dim,
    _resolve_dsv4_kv_cache_dtype,
    _use_v41_mxfp8_kv_record,
)
from vllm.models.deepseek_v41.common.ops import indexer_k_norm_rope_store
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4

LATENT_KEY = "v41_latent_{}"
KPRE_KEY = "v41_kpre_{}"
TOPK_KEY = "v41_topk"
CAND_KEY = "v41_cand"


@dataclass
class StageRelay:
    """What one pipeline stage mirrors, receives and sends."""

    # Upstream KV sources whose compressed cache this stage mirrors.
    kv_mirrors: list[int] = field(default_factory=list)
    # Upstream KV sources whose indexer K cache this stage mirrors.
    k_mirrors: list[int] = field(default_factory=list)
    # Keys arriving from the previous stage / leaving for the next one.
    recv_latents: list[int] = field(default_factory=list)
    recv_kpres: list[int] = field(default_factory=list)
    recv_topk: bool = False
    recv_cand: bool = False
    send_latents: list[int] = field(default_factory=list)
    send_kpres: list[int] = field(default_factory=list)
    send_topk: bool = False
    send_cand: bool = False

    @property
    def active(self) -> bool:
        return bool(
            self.kv_mirrors
            or self.k_mirrors
            or self.recv_latents
            or self.recv_kpres
            or self.recv_topk
            or self.recv_cand
            or self.send_latents
            or self.send_kpres
            or self.send_topk
            or self.send_cand
        )


def plan_stage_relays(config, bounds: list[tuple[int, int]]) -> list[StageRelay]:
    """Per-stage relay plan for the layer ranges ``bounds`` (one per PP rank)."""
    ratios = list(config.compress_ratios)
    kv_sources = sorted(getattr(config, "kv_source_layer_ids", None) or ())
    index_sources = sorted(getattr(config, "index_source_layer_ids", None) or ())
    cand_source = getattr(config, "candidate_source_layer_id", -1)
    has_cand = cand_source >= 0 and getattr(config, "candidate_topk_blocks", 0) > 0

    def kv_source(layer: int) -> int:
        return max(s for s in kv_sources if s <= layer)

    def index_source(layer: int) -> int:
        return max(s for s in index_sources if s <= layer)

    plans = [StageRelay() for _ in bounds]
    needs = []
    for (start, end), plan in zip(bounds, plans):
        layers = range(start, end)
        compressed = [i for i in layers if ratios[i] > 0]
        plan.kv_mirrors = sorted({kv_source(i) for i in compressed if kv_source(i) < start})
        plan.k_mirrors = sorted(
            {
                kv_source(i)
                for i in compressed
                if i in index_sources and i not in kv_sources and kv_source(i) < start
            }
        )
        topk = any(
            i not in index_sources and index_source(i) < start for i in compressed
        )
        cand = has_cand and any(
            i in index_sources and cand_source < i and cand_source < start
            for i in compressed
        )
        needs.append((topk, cand))

    # Walk back from the last stage: what a stage must receive is what it needs
    # itself plus what later stages need and it does not produce.
    recv_l: set[int] = set()
    recv_k: set[int] = set()
    recv_t = recv_c = False
    for r in range(len(bounds) - 1, -1, -1):
        start, end = bounds[r]
        plan = plans[r]
        plan.send_latents, plan.send_kpres = sorted(recv_l), sorted(recv_k)
        plan.send_topk, plan.send_cand = recv_t, recv_c
        layers = range(start, end)
        topk, cand = needs[r]
        recv_l = {s for s in set(plan.kv_mirrors) | recv_l if s < start}
        recv_k = {s for s in set(plan.k_mirrors) | recv_k if s < start}
        recv_t = topk or (recv_t and not any(i in index_sources for i in layers))
        recv_c = cand or (recv_c and cand_source not in layers)
        plan.recv_latents, plan.recv_kpres = sorted(recv_l), sorted(recv_k)
        plan.recv_topk, plan.recv_cand = recv_t, recv_c
    assert not plans[0].recv_latents and not plans[0].recv_kpres
    return plans


class DeepseekV4MirrorKVCache(nn.Module, AttentionLayerBase):
    """Compressed-KV cache of an upstream KV source layer, kept on this stage.

    Registers under the source attention layer's name with the same spec, so
    consumer layers resolve it exactly like the real source.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        attn_cls: type[DeepseekV4Attention],
        layer_id: int,
        prefix: str,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        self.prefix = prefix
        self.layer_id = layer_id
        self.is_kv_source = True
        self.compress_ratio = int(config.compress_ratios[layer_id])
        self.head_dim = config.head_dim
        self.backend_cls = attn_cls.backend_cls
        self.use_fp8_ds_mla_layout = attn_cls.use_fp8_ds_mla_layout
        # Same resolution as DeepseekV4Attention.__init__, so the specs match.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            attn_cls._uses_fp8_ds_mla_layout(self),
            cache_config.cache_dtype,
            cache_config,
        )
        kv_mxfp8 = _use_v41_mxfp8_kv_record()
        self.kv_bytes_per_token = 528 if kv_mxfp8 else 584
        self.kv_page_alignment = 512 if kv_mxfp8 else 576
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            # Plain per-tensor fp8 rows (FlashInfer) carry a unit scale.
            self.register_buffer(
                "_flashinfer_fp8_kv_scale",
                torch.tensor([1.0], dtype=torch.float32),
                persistent=False,
            )
        self.kv_cache = torch.tensor([])
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    get_kv_cache_spec = DeepseekV4Attention.get_kv_cache_spec

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache.squeeze(1)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def forward(self): ...

    def write(self, latent: torch.Tensor, positions: torch.Tensor, attn_metadata):
        """Insert the source stage's latent rows at this step's slots."""
        fp8_scale = getattr(self, "_flashinfer_fp8_kv_scale", None)
        rope_quant_insert(
            latent,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.kv_cache,
            attn_metadata[self.prefix].slot_mapping,
            self.compress_ratio,
            fp8_scale=fp8_scale if self.kv_cache.dtype == torch.float8_e4m3fn else None,
        )


class DeepseekV4MirrorIndexK(nn.Module):
    """Indexer K cache of an upstream KV source layer, kept on this stage.

    Holds the source indexer's ``k_norm`` weight (loaded from the checkpoint by
    ``DeepseekV4Model.load_weights``) to redo ``k_norm -> RoPE -> quant`` on the
    exported ``wk`` projection.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_id: int,
        k_cache_prefix: str,
        rotary_emb: nn.Module,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.compress_ratio = int(config.compress_ratios[layer_id])
        self.use_fp4_kv = dsa_indexer_uses_fp4(vllm_config)
        self.eps = config.rms_norm_eps
        self.rotary_emb = rotary_emb
        self.register_buffer(
            "k_norm_weight",
            torch.ones(config.index_head_dim, dtype=vllm_config.model_config.dtype),
            persistent=False,
        )
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=_indexer_k_cache_head_dim(config.index_head_dim, self.use_fp4_kv),
            dtype=torch.uint8,
            prefix=k_cache_prefix,
            cache_config=vllm_config.cache_config,
            compress_ratio=self.compress_ratio,
        )

    def write(self, k_pre: torch.Tensor, positions: torch.Tensor, attn_metadata):
        indexer_k_norm_rope_store(
            k_pre,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.k_norm_weight,
            self.eps,
            self.k_cache.kv_cache,
            attn_metadata[self.k_cache.prefix].slot_mapping,
            self.compress_ratio,
            self.use_fp4_kv,
        )


def relay_tensor_shapes(config, plan: StageRelay, recv: bool):
    """(key, row shape, dtype) of every relayed tensor a stage receives or sends."""
    latents = plan.recv_latents if recv else plan.send_latents
    kpres = plan.recv_kpres if recv else plan.send_kpres
    shapes = [(LATENT_KEY.format(s), (config.head_dim,), torch.bfloat16) for s in latents]
    shapes += [
        (KPRE_KEY.format(s), (config.index_head_dim,), torch.bfloat16) for s in kpres
    ]
    if plan.recv_topk if recv else plan.send_topk:
        shapes.append((TOPK_KEY, (config.index_topk,), torch.int32))
    if plan.recv_cand if recv else plan.send_cand:
        shapes.append((CAND_KEY, (config.candidate_topk_blocks,), torch.int32))
    return shapes

