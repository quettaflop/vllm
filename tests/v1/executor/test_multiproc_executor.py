# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import FutureWrapper, WorkerProc


def test_future_poll_preserves_rpc_order_without_waiting():
    replies: deque[str] = deque()
    futures: deque[FutureWrapper] = deque()
    first = FutureWrapper(
        futures, replies.popleft, response_ready=lambda: bool(replies)
    )
    second = FutureWrapper(
        futures, replies.popleft, response_ready=lambda: bool(replies)
    )
    assert not second.poll()
    replies.append("execute")
    assert not second.poll()
    assert first.result() == "execute"
    replies.append("sample")
    assert second.poll()
    assert second.result() == "sample"
    assert not futures
    assert second.poll()


def test_future_poll_preserves_worker_exception():
    def fail():
        raise RuntimeError("worker failure")

    future = FutureWrapper(deque(), fail, response_ready=lambda: True)
    assert future.poll()
    with pytest.raises(RuntimeError, match="worker failure"):
        future.result()


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"
