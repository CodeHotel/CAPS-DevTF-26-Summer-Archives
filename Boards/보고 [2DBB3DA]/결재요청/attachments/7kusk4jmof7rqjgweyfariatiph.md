# HAMi Fractional-GPU Feasibility Verification — Handover & Plan

**For:** the agent working on the beta server (single RTX 4080 SUPER, 16 GB)
**From:** a session that ran the design investigation this verification is meant to test
**Nature of this document:** a plan, not a specification. Purpose and goals are fixed; how you structure the code and run the steps is yours to decide.

---

## 1. Background — why this is being done

A student TF at Dongguk University is building a **campus GPU sharing platform**. Labs buy their own GPUs, which then sit idle nights and weekends while other labs have none. The platform pools registered cards so idle capacity becomes usable, with the owning lab keeping priority.

A formal review compared two ways to build it:

- **Job scheduling (Slurm)** — users submit batch jobs, the scheduler orders them, jobs run on whole GPUs.
- **Virtual instance leasing (containers)** — users get a running environment with a GPU slice in it and work interactively.

**The second was chosen.** The deciding reason is granularity: Slurm's smallest GPU allocation unit is one whole card. With roughly a dozen donated cards and forty students in one ML lab session, whole-card allocation cannot serve the class — not with more time or more configuration. Fractional allocation can. It also lets a course guarantee every student an identical enforced environment, which matters when work is graded.

That decision rests on one assumption nobody has tested on real hardware:

> **One physical GPU can be split into several isolated slices, each with an enforced VRAM ceiling, such that real deep-learning workloads run correctly inside a slice and cannot harm a co-tenant.**

The mechanism is **HAMi-core**, which is CNCF-incubating but still relatively young, and is normally deployed under Kubernetes rather than standalone. Its maturity for our use is exactly what is uncertain.

**This verification exists to find out whether that assumption holds.** If it does not, the platform's chosen design is wrong and the team needs to know now rather than in April.

---

## 2. Purpose

Verify, on real hardware, that fractional GPU allocation via HAMi-core is safe and correct enough to build a multi-tenant campus platform on.

A negative result is a successful outcome of this task. Do not tune parameters to make things pass — the point is to learn what is true.

---

## 3. Goal — what exists when this is finished

Two things.

**A findings report** that can state, with measurements behind every clause, whether slices are enforced, whether confined workloads compute correctly, what sharing costs in throughput, and what breaks.

**A working backend runner** — the piece that accepts a job description, allocates a slice, launches the container with the right limits, watches it, records what it actually used, and reclaims capacity when it exits. No REST API, no web UI, no scheduler, no accounting. Just the layer those would eventually sit on top of, structured so they can.

If the verification passes, this runner is the starting point of the real system rather than throwaway scaffolding. Write it accordingly.

---

## 4. What is being tested

**HAMi-core** (`github.com/Project-HAMi/HAMi-core`) builds to `libvgpu.so`, injected with `LD_PRELOAD`. It intercepts calls between the CUDA Runtime and the CUDA Driver.

Facts worth knowing before you start:

- **Memory limiting** hooks driver allocation calls and refuses allocations past a ceiling. Per-process usage is tracked in shared state so multiple processes are meant to share one budget.
- **NVML is hooked too**, so anything querying device memory *inside* a slice sees a virtualized figure. This means in-container NVML is not ground truth — measure from the host.
- **Compute limiting (`CUDA_DEVICE_SM_LIMIT`) is not a hardware partition.** It is software rate limiting using nanosleep delays on kernel launches. Expect non-linear, workload-dependent behavior quite unlike MIG.
- **No kernel programming, no driver patching, no application changes.** It is userspace.

Documented variables are `CUDA_DEVICE_MEMORY_LIMIT` (accepts `4g`, `4096m`, or bytes), `CUDA_DEVICE_SM_LIMIT` (integer percent), and `LIBCUDA_LOG_LEVEL` (0–4). Get the authoritative list out of the source rather than trusting this — grepping for `getenv` in the repo is the fast way, and per-device forms may exist that are not documented.

**One known incompatibility to confirm rather than discover:** HAMi issue #1360 reports PyTorch CUDA Graphs failing under HAMi, and a matching vLLM issue reports memory clamps being ignored during CUDA-graph profiling — filed against HAMi, MIG and MPS together, so it is a property of memory-clamped GPUs generally. Characterize it; don't try to fix it.

---

## 5. Verification phases

Each phase answers a question. How you implement it is your call.

**A. Environment and instrument.** Capture the exact stack. Then validate your measurement tooling against a known allocation before trusting it — if a script allocates exactly 2 GB and your sampler disagrees, everything after is noise. Also establish how reproducible each workload is when run twice unconfined, since that sets the tolerance for judging confined runs later.

**B. Native baselines.** Run a set of real workloads unconfined and record peak VRAM and final results. These are the reference everything is compared against. Also note *when* peak memory arrives during a run — early peaks mean a bad fit can be caught and killed quickly, late peaks mean the platform must reserve for the worst case up front.

**C. Build and smoke test.** Build `libvgpu.so`, pin the commit. Then run the control that makes everything else meaningful: preload the library with *no* limits set and confirm nothing changes. If merely being intercepted breaks correctness or costs real performance, the approach is unsafe under every user job and the verification stops there.

**D. Enforcement.** Is the ceiling real, is it where it claims to be, and what happens when a job hits it? Probe the actual usable ceiling against the configured one; the gap is fixed overhead and you need its size. Measure what an additional container costs even when idle, because that overhead multiplied by slice count is capacity the platform never gets to sell. Confirm that overrunning produces a catchable OOM rather than a driver fault.

**E. Real workloads under a ceiling.** Do the workloads from phase B produce the same results when confined? Then find the minimum headroom above measured peak that still works reliably — that number becomes the platform's sizing rule. Also separate the cost of being *intercepted* from the cost of being *constrained*.

**F. Co-tenancy.** The phase this whole exercise exists for. Two containers on one card: do both compute correctly, how much does each slow down, and does running them together actually finish both sooner than running them back to back? Then the safety test — one tenant behaving adversarially (allocating until OOM in a loop, long kernels, abrupt kill) must not corrupt, kill, or fault the other. Being *slowed* by a co-tenant is contention and fine; being *wrong* is an isolation breach and fatal to the design.

Also confirm the ceiling binds per *slice*, not per *process*. If a user can bypass the limit by forking, the limit is decorative.

**G. Failure modes.** Reproduce the CUDA Graphs incompatibility and document which framework paths it affects. Then the operational cases: hard-killing a container mid-training, rapid create/destroy cycles, distinguishing a container OOM-kill from a CUDA OOM. These decide what the platform has to handle and what it has to tell users.

**H. The runner.** Assemble everything learned into the backend described in §3, then exercise it — submit a batch of jobs, run them two at a time, confirm every job completes, every record is accurate, and capacity returns to full afterward. A few hours of continuous operation with no leaked memory or stranded capacity is the bar.

---

## 6. Stop conditions

Halt and report rather than continuing, if any of these occur:

- Preloading the library with no limits set changes results or costs significant performance.
- An allocation succeeds past the configured ceiling.
- An in-slice OOM produces a host-level GPU fault, Xid error, or driver reset.
- A confined workload computes a materially different result than its unconfined baseline.
- A co-tenant can corrupt, kill, or fault another tenant.
- The memory ceiling binds per process rather than per slice.

Each of these invalidates the phases after it, and several invalidate the platform design itself.

---

## 7. Out of scope

REST API, web UI, authentication. Multi-node anything. Scheduling, queueing, fair-share policy. Credits, tiers, pricing — those wait on administrative decisions the team does not have yet. Performance *prediction* — a separate research question, and this task measures rather than forecasts. Storage sync, SSH routing, reverse proxying. And Kubernetes: HAMi normally ships as a k8s device plugin and we are deliberately using it standalone under plain Docker, so if you find yourself installing k8s, stop.

---

## 8. Choosing workloads

Use well-known model/dataset pairs with published reference numbers, so "the model is learning correctly" is checkable rather than assumed. Measure each one's peak VRAM unconfined first, then run it inside a slice sized from that measurement.

Pick for **variety of memory shape**, not variety of model name — six CNNs would test one thing six times. Cover at least: something small and activation-dominated (ResNet-18 on CIFAR-10), something large and activation-dominated (ResNet-50 on an ImageNet subset), something attention-based and parameter-heavy (BERT on a GLUE task), something autoregressive (GPT-2 on WikiText), and at least one **inference** workload with a multi-module load pattern (Stable Diffusion), since inference allocates very differently from training and is a likely place for slicing to break.

Two synthetic tools are worth having as well: one that allocates in fixed steps until it fails, for probing ceilings precisely, and one that is pure compute, for the SM-limit sweep.

Size everything for a 16 GB card. Slices worth testing are roughly 2, 4, and 8 GB. Download all datasets and weights once up front so nothing pulls from the network during a timed run.

---

## 9. Reporting back

Keep a running log as you go rather than reconstructing at the end.

The report should state what was verified, what failed, and what the numbers were — including the ones that came out worse than hoped. Two figures in particular will be used directly in platform design decisions and should be called out plainly: **the minimum headroom multiplier over measured peak**, and **the per-container fixed overhead**, since the second determines how many slices a card can actually hold once you subtract it N times.

Anything you had to do differently from this plan should be recorded with the reason. This document was written without access to the machine and will be wrong in places.
