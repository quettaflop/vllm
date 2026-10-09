# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.executor.multiproc_executor import FutureWrapper


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("coalesce", [False, True])
def test_completed_batches_drain_before_rescheduling(enabled, coalesce):
    """Ready groups merge only after every available output reaches scheduling."""
    queue: deque[tuple[Future[int], str, Future[int]]] = deque()
    outputs = []
    for i in range(2):
        future: Future[int] = Future()
        if coalesce:
            future = FutureWrapper(deque(), lambda i=i: i, response_ready=lambda: False)
        else:
            future.set_result(i)
        queue.appendleft((future, f"batch{i}", future))
        outputs.append({i: object()})
    pending: Future[None] = Future()
    scheduler = Mock()
    scheduler.should_defer_pp_rebatch.return_value = coalesce
    scheduler.schedule.return_value = SimpleNamespace(total_num_scheduled_tokens=0)
    scheduler.update_from_output.side_effect = outputs
    executor = Mock()
    executor.has_ready_output.side_effect = lambda future: future.done()
    executor.execute_model.return_value = pending
    engine = SimpleNamespace(
        batch_queue=queue,
        batch_queue_size=5,
        pp_ready_rebatch=enabled,
        scheduler=scheduler,
        model_executor=executor,
        is_ec_consumer=True,
        is_pooling_model=False,
        _pp_trace_steps=None,
        _should_throttle_prefills=lambda: False,
        capture_iteration_details=lambda _: nullcontext(),
        log_error_detail=lambda _: nullcontext(),
        _process_aborts_queue=Mock(),
        _attach_iteration_details=Mock(),
    )
    if enabled:
        assert EngineCore.step_with_batch_queue(engine) == (outputs[0], False)
        assert EngineCore.step_with_batch_queue(engine) == (outputs[1], False)
        scheduler.schedule.assert_not_called()
        assert not queue
        engine._process_aborts_queue.assert_called()
    else:
        assert EngineCore.step_with_batch_queue(engine) == (None, False)
        assert len(queue) == 3
        scheduler.update_from_output.assert_not_called()
    assert EngineCore.step_with_batch_queue(engine) == (None, False)
    scheduler.schedule.assert_called()


@pytest.mark.parametrize(
    "enabled, outputs, sleeps",
    [
        (True, {}, False),
        (True, None, True),
        (False, {}, True),
    ],
)
def test_output_draining_does_not_sleep(monkeypatch, enabled, outputs, sleeps):
    sleep = Mock()
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", sleep)
    engine = SimpleNamespace(
        step_fn=lambda: (outputs, False),
        output_queue=Mock(),
        post_step=Mock(),
        scheduler=Mock(),
        pp_ready_rebatch=enabled,
    )
    assert not EngineCoreProc._process_engine_step(engine)
    assert sleep.called == sleeps
