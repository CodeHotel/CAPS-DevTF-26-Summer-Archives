# HAMi Verification — Third Round (task_handout_3)

**To:** the agent that produced the round 1 and round 2 findings reports.
**Nature:** a plan, same register as the previous two. Purpose and goals are fixed; method is yours.
**Prerequisite:** both prior reports. Nothing settled in either is revisited.

---

## 1. Where things stand

Both rounds were accepted. Slicing is real, correctness holds bitwise, isolation holds under adversarial neighbours, eight to nine tenants share a card with slowdown tracking 1/N, the tier ladder is derived and confirmed, non-root tenants work, and the calibration routine covers per-card constants.

The remaining uncertainty is nearly exhausted. This round is small — **three items**, chosen by one test: *does a negative result change the plan?* Everything else that was still open failed that test and has been closed by decision rather than experiment:

- **Display-attached cards** — a node admission policy. Require headless or size conservatively.
- **Storage under concurrent load** — premature; no storage architecture has been chosen yet.
- **Old cards** — resolved into a planning decision. HAMi's own matrix lists NVIDIA support as "All" with only `driver >= 440`, so HAMi is not the constraint. CUDA 13 dropping Maxwell/Pascal/Volta is, and the question is simply whether pre-Turing cards are accepted into the pool at the cost of a second CUDA 12.x image line.
- **Static CUDA binaries** — unfixable; the output is a documented threat model regardless of any test.
- **Session restart with persistence** — ordinary development testing. A failure is a bug to fix, not a design fork.

---

## 2. Task A — multi-GPU jobs under HAMi

**Why it matters.** Every result so far is one container, one GPU. Some donated machines will have two or four cards, and research jobs that need more than one exist. If distributed training does not work inside a preloaded container, multi-GPU jobs must bypass the slicing layer entirely — two allocation paths instead of one, which is an architectural fork worth knowing about now.

**A hardware caveat to handle honestly.** The verification box has one GPU, so parts of this cannot be answered there. Say clearly in the report which questions you answered and which you could not.

**Answerable on one card:**

- Whether a HAMi-preloaded container can initialize NCCL at all, and whether the process group forms.
- **Whether NCCL's own buffers are counted against the slice.** NCCL allocates communication buffers outside PyTorch's caching allocator, and newer versions use `cuMemCreate`/`cuMemMap` rather than plain `cuMemAlloc`. Round 1 found `cuMemCreate` is hooked — confirm that NCCL's allocations actually land inside the ceiling rather than escaping it. A leak here would mean a multi-GPU tenant can exceed its declared slice, which matters more than whether the job runs.
- Whether `gloo` behaves differently from `nccl`, to separate "interception breaks collectives" from "interception breaks NCCL specifically."

**Needs two cards:**

- Whether one container can hold slices on two distinct physical devices, and whether `CUDA_DEVICE_MEMORY_LIMIT_1` is genuinely independent of `_0` (only the `_0` form has ever been exercised).
- Whether real DDP training across two sliced GPUs completes and produces correct results.

If no second GPU is reachable, an hour on a cheap two-card cloud instance would settle it — pick anything Turing or newer so the CUDA 13 stack applies unchanged. The question is whether interception breaks collectives, which is not especially card-specific.

**The decision this feeds.** If multi-GPU under slicing does not work or cannot be verified, the fallback is that multi-GPU jobs receive whole cards with no slicing, on a separate path. That is simple and safe, and it costs only the ability to fractionally share a machine that is running a distributed job. Frame the result against that fallback so it produces an answer either way.

---

## 3. Task B — long idle holds

**Why it matters.** This is the platform's primary usage pattern and the only shape never tested. Every measurement so far has been short jobs or three hours of churn. Real leasing means a session held eight to twelve hours, mostly idle, with bursts of work — a student leaves Jupyter open over lunch, a researcher holds a session overnight.

**What to find out.** Whether anything drifts over a long hold: HAMi's shared accounting state, its utilisation-watcher thread, host-visible VRAM, container RSS and CPU. Then whether the slice still behaves after the hold — that the ceiling still enforces at the same value, and that a workload run at hour twelve produces the same bitwise result as one run at minute one.

**Include a bursty pattern, not just flat idle.** Alternating work and idle is what actually happens, and it exercises allocate/free cycles across the hold rather than a single steady state. A long-lived Jupyter kernel with intermittent GPU use is a fair model.

**The decision this feeds.** A negative result forces session time caps or periodic recycling, both of which are user-visible policy and both of which undercut the "leave it running" convenience that motivated choosing leasing in the first place. So the useful output is not just pass or fail but, if something does drift, **how long a session can safely be held.**

Mostly unattended. Twelve hours of wall time, very little of it yours.

---

## 4. Task C — devel image and CUDA extensions

**Why it matters.** The runtime image has no `nvcc`, so a tenant cannot compile a CUDA extension. Coursework does not care; research users do routinely — custom kernels, flash-attention variants, apex, xformers. This decides whether the platform can serve them at all.

**What to find out.** Build a devel-based variant of the workload image and repeat the round 2 non-root conditions on it: uid 1000, read-only rootfs, HOME on an exec-capable volume, `/etc/ld.so.preload` bound read-only. Then confirm a tenant can actually compile and run something — `torch.utils.cpp_extension` with a trivial CUDA kernel is enough and takes seconds; a full flash-attention build is not necessary to answer the question.

Two things to check alongside: that a compiled extension runs correctly **under the slice** and respects the ceiling, and the image size delta, since every node has to hold whatever images you distribute.

**Also confirm the ceiling still holds on the devel image.** A devel image carries a fuller CUDA toolchain, and it is worth one check that none of the round 1 bypass routes reopen there.

**The decision this feeds.** Either the platform offers a second image tier for research users, or it tells them to compile elsewhere and bring a wheel. Likely to pass, cheap to establish, and awkward to discover late.

---

## 5. Reporting

Same discipline. A negative result is a successful outcome — particularly for Task A, where a failure redirects architecture rather than blocking it.

Three sentences will be used directly: **whether NCCL allocations stay inside the slice**, **the longest safe session hold**, and **whether a non-root tenant can compile a CUDA extension.** Everything else supports those.
