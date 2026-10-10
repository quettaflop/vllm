# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer PCIe CUDA-IPC all-reduce integration.

This backend targets small tensor-parallel all-reduces on a single PCIe-only
node.  Its workspace has a stricter lifetime and stream contract than the
existing FlashInfer MNNVL/TRT-LLM all-reduce implementation, so it intentionally
lives behind a separate wrapper and an opt-in environment variable.

The kernels spin on peer flags with no timeout and no metadata exchange, so
every rank must issue the same sequence of collective calls.  A rank that
quietly falls back to another backend while its peers still use the IPC path
leaves the group spinning, so this wrapper disables the backend group-wide
(never on a single rank) whenever any rank cannot support it.
"""

import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import import_pynvml

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _collective_disable_reason(rank_errors: Sequence[str | None]) -> str | None:
    """Collapse per-rank local failures into one group-wide disable reason.

    The IPC kernels have no timeout, so a single rank dropping out while its
    peers keep going hangs the group.  Every rank must reach the same decision;
    this turns the gathered per-rank reasons into a single message so they all
    disable the backend together.

    Args:
        rank_errors: Local failure reason for each group rank, or None when
            that rank can use the backend.

    Returns:
        None when every rank is ready, otherwise a message naming the ranks
        that failed and why.

    """
    failed = [
        f"rank {rank} ({error})"
        for rank, error in enumerate(rank_errors)
        if error is not None
    ]
    if not failed:
        return None
    return (
        "FlashInfer PCIe IPC all-reduce is unavailable on "
        f"{len(failed)}/{len(rank_errors)} ranks; disabling it group-wide: "
        + ", ".join(failed)
    )


def _island_placement_error(
    numa_nodes: Sequence[int | None], world_size: int
) -> str | None:
    """Check the 4+4 island rule the copy-engine schedule assumes.

    FlashInfer's ``COPY_ENGINE_ISLAND`` decomposition assigns island 0 to ranks
    0-3 and island 1 to ranks 4-7, so the grouping only describes the fabric
    when each half is co-socket.  It is reachable at world size 8 only (see
    ``flashinfer.comm.pcie_ipc_policy``).  A single NUMA/socket domain needs no
    split, and an undeterminable mapping is left alone rather than guessed at.

    Args:
        numa_nodes: NUMA node id of each rank's GPU, or None when it could not
            be determined.
        world_size: Size of the process group.

    Returns:
        A message describing the violation, or None when the placement is
        acceptable or cannot be judged.

    """
    if world_size != 8 or any(node is None for node in numa_nodes):
        return None
    if len(set(numa_nodes)) < 2:
        return None
    if len(set(numa_nodes[:4])) == 1 and len(set(numa_nodes[4:])) == 1:
        return None
    return (
        "FlashInfer PCIe IPC all-reduce needs the 4+4 island decomposition to "
        "align with NUMA/socket domains, but ranks 0-3 map to NUMA nodes "
        f"{list(numa_nodes[:4])} and ranks 4-7 to {list(numa_nodes[4:])}"
    )


def _local_numa_node(device: torch.device) -> int | None:
    """NUMA node hosting this rank's GPU, or None when NVML cannot report it."""
    try:
        device_index = device.index
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        uuid = current_platform.get_device_uuid(device_index)
        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            return int(pynvml.nvmlDeviceGetNumaNodeId(handle))
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - unknown placement stays with another backend
        return None


try:
    import flashinfer.comm as flashinfer_comm

    _pcie_ipc_available = hasattr(flashinfer_comm, "PcieIpcAllReduceWorkspace")
except ImportError:
    flashinfer_comm = None  # type: ignore[assignment]
    _pcie_ipc_available = False


class FlashInferPcieIpcAllReduce:
    """vLLM lifecycle wrapper for FlashInfer's PCIe IPC all-reduce.

    The underlying workspace is a strict collective: every rank must issue the
    same sequence of calls with the same shape, dtype and launch configuration,
    and a rank that falls back on its own hangs the group.  This wrapper makes
    that fallback group-wide instead.

    The workspace also serves a single CUDA stream.  Its epoch and arrival
    counters assume the calls sharing it are totally ordered, which stream
    order gives and concurrent streams do not; build one instance, and hence
    one workspace, per stream rather than sharing one across streams.
    """

    def __init__(
        self,
        group: ProcessGroup,
        tune_group: ProcessGroup,
        device: int | str | torch.device,
    ) -> None:
        self.disabled = True
        self.group = group
        self.tune_group = tune_group
        self.device = torch.device(device)
        self.world_size = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.workspace: Any | None = None
        self.hidden_dim = 0
        self.dtype: torch.dtype | None = None

        # Resolve the local reasons before the exchange below, which every rank
        # must enter so the group agrees on one decision.
        reason = self._local_support_error()
        reason = self._collective_support_error(reason)
        if reason is not None:
            logger.warning_once(
                "FlashInfer PCIe IPC all-reduce was requested but is disabled "
                "on every rank: %s. Falling back to another all-reduce backend.",
                reason,
            )
            return

        # Setup is deferred until kernel_warmup, where the model hidden size and
        # exact CUDA Graph buckets are known. Until then dispatch falls through.
        self.disabled = False

    def _local_support_error(self) -> str | None:
        """Why this rank alone cannot use the backend, or None if it can."""
        if not _pcie_ipc_available:
            return "this FlashInfer build does not provide PcieIpcAllReduceWorkspace"
        if not current_platform.is_cuda():
            return "it requires the CUDA platform"
        if self.world_size not in _SUPPORTED_WORLD_SIZES:
            return (
                f"it does not support world_size={self.world_size} "
                f"(supported: {_SUPPORTED_WORLD_SIZES})"
            )
        return None

    def _collective_support_error(self, local_error: str | None) -> str | None:
        """Agree group-wide on whether the backend can be used.

        Every rank calls this with its own local result, so the exchange and
        the topology checks below are entered by the whole group.  A failure on
        any rank disables the backend on all of them, which an asymmetric
        fallback would not.
        """
        errors: list[str | None] = [None] * self.world_size
        dist.all_gather_object(errors, local_error, group=self.group)
        group_error = _collective_disable_reason(errors)
        if group_error is not None:
            return group_error

        if not all(in_the_same_node_as(self.tune_group, source_rank=0)):
            return "the TP group is not on a single node"

        if self.world_size == 8:
            numa_nodes: list[int | None] = [None] * self.world_size
            dist.all_gather_object(
                numa_nodes, _local_numa_node(self.device), group=self.group
            )
            placement_error = _island_placement_error(numa_nodes, self.world_size)
            if placement_error is not None:
                return placement_error
        return None

    @property
    def initialized(self) -> bool:
        return not self.disabled and self.workspace is not None

    def setup(
        self,
        *,
        hidden_dim: int,
        dtype: torch.dtype,
        capture_sizes: Sequence[int],
        tune_cache: Path,
    ) -> None:
        """Allocate, tune, and prepare the exact graph-capture shapes."""
        if self.disabled or self.workspace is not None:
            return
        if dtype not in _SUPPORTED_DTYPES:
            logger.warning_once(
                "FlashInfer PCIe IPC all-reduce does not support dtype=%s; "
                "falling back to another all-reduce backend.",
                dtype,
            )
            self.disabled = True
            return

        batches = tuple(sorted({int(size) for size in capture_sizes if size > 0}))
        # VLLM_FI_PCIE_IPC_PREFILL_TOKENS=N extends the workspace to N-token
        # all-reduces (the prefill chunk) and tunes a ladder of prefill sizes,
        # so chunked-prefill all-reduces take the IPC path instead of NCCL.
        # Sizes between ladder rungs resolve lazily (one collective agreement
        # per new size, outside graph capture), as FlashInfer documents.
        prefill_tokens = int(os.environ.get("VLLM_FI_PCIE_IPC_PREFILL_TOKENS", "0"))
        if prefill_tokens > (batches[-1] if batches else 0):
            ladder = [
                b
                for b in (
                    640,
                    768,
                    1024,
                    1536,
                    2048,
                    3072,
                    4096,
                    6144,
                    8192,
                    12288,
                    16384,
                )
                if (batches[-1] if batches else 0) < b < prefill_tokens
            ]
            batches = tuple(sorted(set(batches) | set(ladder) | {prefill_tokens}))
        if not batches:
            logger.warning_once(
                "FlashInfer PCIe IPC all-reduce has no CUDA Graph capture sizes "
                "to prepare; falling back to another all-reduce backend."
            )
            self.disabled = True
            return

        self.hidden_dim = int(hidden_dim)
        self.dtype = dtype
        self.min_tokens = int(os.environ.get("VLLM_FI_PCIE_IPC_MIN_TOKENS", "0"))
        # VLLM_FI_PCIE_IPC_SMALL_TOKENS=N also routes all-reduces of at most N
        # tokens (single-request decode) through IPC, below the NCCL band.
        self.small_tokens = int(os.environ.get("VLLM_FI_PCIE_IPC_SMALL_TOKENS", "0"))
        max_numel = batches[-1] * self.hidden_dim
        workspace = flashinfer_comm.PcieIpcAllReduceWorkspace(
            group=self.group,
            max_numel=max_numel,
            dtype=dtype,
            tune_batches=batches,
            tune_cache=str(tune_cache),
        )
        self.workspace = workspace

        # tune() reuses a complete persisted cache and profiles only cache
        # misses. It is deliberately mandatory: FlashInfer's seed policy was
        # substantially slower than NCCL for the target TP4 decode workload.
        torch.accelerator.synchronize(self.device)
        workspace.rebind_stream()
        workspace.tune(
            [self.hidden_dim],
            dtype=dtype,
            tune_group=self.tune_group,
        )
        workspace.prepare([(batch, self.hidden_dim) for batch in batches], dtype=dtype)
        if prefill_tokens > 0:
            # Resolve every token count up to the prefill limit now. A size seen
            # for the first time at serve time costs one collective plus a host
            # readback, which drains the GPU queue and breaks vLLM's CPU/GPU
            # overlap; eager prefill steps hit a new size almost every step.
            import time as _time

            t0 = _time.perf_counter()
            done = set(batches)
            workspace.prepare(
                [
                    (b, self.hidden_dim)
                    for b in range(1, prefill_tokens + 1)
                    if b not in done
                ],
                dtype=dtype,
            )
            logger.info_once(
                "FlashInfer PCIe IPC: pre-resolved %d sizes up to %d tokens in %.1f s",
                prefill_tokens - len(done & set(range(1, prefill_tokens + 1))),
                prefill_tokens,
                _time.perf_counter() - t0,
            )
        torch.accelerator.synchronize(self.device)
        workspace.rebind_stream()
        logger.info_once(
            "Initialized FlashInfer PCIe IPC all-reduce for TP%d, hidden_dim=%d, "
            "dtype=%s, max_tokens=%d.",
            self.world_size,
            self.hidden_dim,
            dtype,
            batches[-1],
        )

    def should_use(self, inp: torch.Tensor) -> bool:
        workspace = self.workspace
        # VLLM_FI_PCIE_IPC_MIN_TOKENS=N keeps all-reduces under N tokens (decode
        # steps, which the NCCL tuner plugin serves better on TP8) on NCCL and
        # sends only prefill-sized ones through IPC.
        return bool(
            not self.disabled
            and workspace is not None
            and inp.is_cuda
            and inp.is_contiguous()
            and inp.dim() == 2
            and (inp.shape[0] >= self.min_tokens or inp.shape[0] <= self.small_tokens)
            and inp.shape[1] == self.hidden_dim
            and inp.dtype == self.dtype
            and workspace.supports(inp)
        )

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        assert self.workspace is not None
        return self.workspace.all_reduce(inp)

    @contextmanager
    def capture(self):
        """Move the single-stream workspace to and from the capture stream."""
        workspace = self.workspace
        if workspace is None:
            yield
            return

        # The caller orders graph capture against normal execution. A device
        # sync makes that assertion explicit before relaxing FlashInfer's
        # single-stream guard. Captured launches themselves are exempt from the
        # guard, while capture warmups execute and bind to the capture stream.
        torch.accelerator.synchronize(self.device)
        workspace.rebind_stream()
        try:
            yield
        finally:
            torch.accelerator.synchronize(self.device)
            workspace.rebind_stream()

    def destroy(self) -> None:
        workspace = self.workspace
        if workspace is not None:
            workspace.destroy()
            self.workspace = None


def warmup_flashinfer_pcie_ipc_allreduce(worker: "Worker") -> None:
    """Initialize the TP PCIe IPC backend immediately before graph capture."""
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.model_executor.warmup.flashinfer_autotune_cache import (
        resolve_flashinfer_autotune_file,
    )

    tp_group = get_tp_group()
    communicator = tp_group.device_communicator
    if not isinstance(communicator, CudaCommunicator):
        return
    pcie_comm = communicator.fi_pcie_ipc_ar_comm
    if pcie_comm is None or pcie_comm.disabled:
        return
    if worker.vllm_config.parallel_config.use_ubatching:
        logger.warning_once(
            "FlashInfer PCIe IPC all-reduce does not yet support DBO or "
            "multi-ubatch execution; falling back to another backend."
        )
        pcie_comm.disabled = True
        return

    capture_sizes = worker.vllm_config.compilation_config.cudagraph_capture_sizes
    if not capture_sizes:
        return

    base_cache = resolve_flashinfer_autotune_file(worker.model_runner)
    ranks = "-".join(str(rank) for rank in tp_group.ranks)
    tune_cache = base_cache.with_name(f"pcie_ipc_allreduce_tp_{ranks}.json")
    pcie_comm.setup(
        hidden_dim=worker.model_config.get_hidden_size(),
        dtype=worker.model_config.dtype,
        capture_sizes=capture_sizes,
        tune_cache=tune_cache,
    )
