# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from collections import deque
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pp_utils


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


@pytest.fixture
def ready_handler(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("PP token delivery requires CUDA streams")
    group = Mock(is_last_rank=False, last_rank=3, world_size=4)
    monkeypatch.setattr(pp_utils, "get_pp_group", lambda: group)
    handler = pp_utils.PPHandler(8, 2, torch.device("cuda"), ready_rebatch=True)
    payloads: deque[torch.Tensor] = deque()

    def broadcast(tensor, **kwargs):
        tensor.copy_(payloads.popleft())

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    def receive(indices, tokens, accepted, rejected, drafts, prefill=None):
        n = len(indices)
        batch = _batch([10] * n, prefill or [8] * n, [3] * n)
        batch.idx_mapping_np = np.asarray(indices, dtype=np.intp)
        batch.idx_mapping = torch.tensor(indices, device="cuda")
        payloads.extend(
            [
                torch.tensor(tokens, dtype=torch.int64, device="cuda"),
                torch.tensor([accepted, rejected], dtype=torch.int32, device="cuda"),
                torch.tensor(drafts, dtype=torch.int64, device="cuda"),
            ]
        )
        handler.receive(batch)
        assert not payloads

    return handler, receive


def test_ready_delivery_merges_steps_and_consumes_each_row_once(ready_handler):
    """Changing batch order must preserve sample, rejection and draft ownership."""
    handler, receive = ready_handler
    drafts = torch.full((8, 2), -1, dtype=torch.int64, device="cuda")
    handler.get_ready_sampled_outputs([])
    receive([0, 1], [[10, 11, -1], [20, -1, -1]], [2, 1], [1, 2], [[12, 13], [21, 22]])
    handler.get_ready_sampled_outputs([])
    receive([2], [[30, 31, 32]], [3], [0], [[33, 34]])

    # A later step is consumed before the remaining row of the earlier step.
    outputs = handler.get_ready_sampled_outputs([2, 0], drafts)
    assert [o["idx_mapping"].tolist() for o in outputs] == [[0], [2]]
    assert [o["sampled_tokens"].tolist() for o in outputs] == [
        [[10, 11, -1]],
        [[30, 31, 32]],
    ]
    assert [o["num_rejected"].tolist() for o in outputs] == [[1], [0]]
    assert drafts[:3].tolist() == [[12, 13], [-1, -1], [33, 34]]
    assert handler.get_ready_sampled_outputs([0, 2], drafts) == []
    remaining = handler.get_ready_sampled_outputs([1], drafts)
    assert remaining[0]["num_sampled"].tolist() == [1]
    assert remaining[0]["num_rejected"].tolist() == [2]
    assert drafts[1].tolist() == [21, 22]


def test_ready_delivery_ignores_freed_and_reused_slots(ready_handler):
    handler, receive = ready_handler
    handler.get_ready_sampled_outputs([])
    receive([0, 1], [[10, -1, -1], [20, -1, -1]], [1, 1], [2, 2], [[11, 12], [21, 22]])
    handler.on_req_idx_freed(0)
    assert handler.get_ready_sampled_outputs([0]) == []
    receive([0], [[30, -1, -1]], [1], [2], [[31, 32]])
    outputs = handler.get_ready_sampled_outputs([0, 1])
    got = {int(o["idx_mapping"].item()): o["sampled_tokens"].tolist() for o in outputs}
    assert got == {0: [[30, -1, -1]], 1: [[20, -1, -1]]}


def test_ready_delivery_excludes_non_final_prefill_rows(ready_handler):
    handler, receive = ready_handler
    handler.get_ready_sampled_outputs([])
    receive(
        [0, 1],
        [[-1, -1, -1], [20, -1, -1]],
        [0, 1],
        [0, 2],
        [[-1, -1], [21, 22]],
        prefill=[100, 8],
    )
    assert handler.get_ready_sampled_outputs([0]) == []
    outputs = handler.get_ready_sampled_outputs([1])
    assert outputs[0]["idx_mapping"].tolist() == [1]
    assert outputs[0]["sampled_tokens"].tolist() == [[20, -1, -1]]
