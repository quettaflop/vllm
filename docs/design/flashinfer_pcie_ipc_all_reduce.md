# FlashInfer PCIe IPC all-reduce

`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` opts the TP group into FlashInfer's
PCIe CUDA-IPC all-reduce for small tensor-parallel collectives on a single
PCIe-only node.

This backend is **already upstream** in vLLM v0.30.0
(`[Distributed] Add opt-in FlashInfer PCIe IPC all-reduce backend (#53576)`), so
this fork carries no port of the transport itself. The pinned FlashInfer
already ships `flashinfer.comm.PcieIpcAllReduceWorkspace` and the surrounding
kernels, so **no rebuild is required to use it**. The default stays off; the
variable is opt-in.

## Enabling

```bash
VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 <your launcher>
```

It is selected for the `tp` group only and is disabled automatically when
`VLLM_BATCH_INVARIANT` is set. Set it uniformly on every rank: enabling it on a
subset of the group means the other ranks never construct the communicator, so
there is no exchange to agree on and initialization hangs.

## Strictness constraints

The workspace is a collective with no timeout and no metadata exchange. Three
properties must hold:

1. **Identical call sequence.** Every rank must issue the same sequence of
   calls with the same shape, dtype and launch configuration, in the same
   order. A rank that skips a call, reorders two, or passes a different
   explicit `config` does not get an error: the group hangs, or a rank reads a
   neighbour's partial sums as if they were finished.
2. **One CUDA stream per workspace.** The epoch and arrival counters assume the
   calls sharing a workspace are totally ordered, which stream order gives and
   concurrent streams do not. One instance (and therefore one workspace) serves
   exactly one stream.
3. **4+4 island placement at world size 8.** The copy-engine island variant
   (`COPY_ENGINE_ISLAND`) is reachable only at world size 8, and its islands are
   assigned by logical rank (`island = rank < 4 ? 0 : 1`), so ranks 0-3 must be
   co-socket and ranks 4-7 co-socket for the schedule to describe the fabric.

Because of (1), a *per-rank* fallback is unsafe: seven ranks using the IPC path
while one drops to NCCL hangs the group. `FlashInferPcieIpcAllReduce` therefore
disables the backend **group-wide** whenever any rank cannot support it, checks
the (3) placement from the GPUs' NUMA nodes, and documents (2) for callers.

## Measured

On gpu07 (8 ranks, real PR kernel, bit-exact vs NCCL, graph-capturable):

- **3.57x eager / 4.03x graph** geomean vs the live-tuner NCCL config for the
  message sizes this service uses.
- The world-8 kernel path was confirmed to run (no fallback to a smaller-world
  kernel).
- The live decode step is 8.6 ms at batch 1, of which NCCL all-reduce is about
  35% across 81 collectives per step.

## Rollback

Unset `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` (or use the previous launcher);
the backend is then never constructed and the group falls back to whatever it
used before. No state is written outside the FlashInfer autotune cache.

## Reachability check

To confirm a built image can reach the backend without starting a server:

```bash
python -c "import flashinfer.comm as c; assert hasattr(c, 'PcieIpcAllReduceWorkspace')"
```

A failure here means the pinned FlashInfer changed shape; check the pin before
enabling the variable.
