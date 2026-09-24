# HAMi Verification — Second Round (task_handout_2)

**To:** the agent that produced `HAMi 기술검증 결과.md` on 2026-09-02.
**Nature:** a plan, same as the first handout. Purpose and goals are fixed; method and structure are yours.
**Prerequisite:** your own first report. This document assumes it and does not repeat it.

---

## 1. Where things stand

The first report was reviewed and accepted. The verdict — that HAMi-core slices are real, that confined workloads compute bitwise-identically, that co-tenants cannot corrupt or fault each other — is taken as established. None of it needs redoing.

Three things about how you worked are worth saying, because they should carry into this round: you validated the sampler against a known allocation before trusting it; you used SHA over the final `state_dict` rather than metric tolerance; and you recorded deviations against your own interest, including the initially-omitted SM sweep. You also corrected a false premise in the first handout — issue #1360 was not a CUDA Graphs failure — and tested the real question directly instead. Keep all of that.

**This round exists because of one gap.** Every co-tenancy result in the report is a **pair**, and the soak ran at `--concurrency 2`. The platform this verification supports is premised on **seven or eight students sharing one card in a lab session**. The claim currently rests on extrapolation from N=2, and two simultaneous CUDA contexts is a different regime from eight — time-slice fairness and context-switch overhead are exactly the things that degrade nonlinearly.

Everything else below is smaller.

---

## 2. Tasks

### A. High-N concurrency — the blocker

Find out whether slowdown stays near 1/N as N grows, or whether something breaks or degrades faster than proportionally.

Run identical workloads at **N = 2, 4, 8** in tight slices and compare per-job slowdown against the solo baseline. ResNet-18 at 1728m is the natural vehicle — eight of them debit 14336 MiB against a 16120 MiB pool, so it fits with room. Include **at least one mixed-size configuration** (one large tenant plus several small ones), because uniform tenants are the easy case and a real lab session will not be uniform.

Watch for: correctness holding at every N (SHA, as before); slowdown tracking 1/N or diverging from it; whether aggregate throughput collapses somewhere between 4 and 8; and any Xid at all.

**Done when** there is a slowdown-vs-N curve with correctness verified at each point, and a defensible statement about the largest N this card supports for coursework-sized work.

### B. Startup storm

A lab session means forty students clicking start within about two minutes. Container churn measured 3.8 s per job at concurrency 1, but simultaneous CUDA initialization was never tested.

Launch **8 containers as close to simultaneously as possible** and measure how long until all are training, whether any fail to initialize, and whether the pool's accounting stays correct when eight allocations arrive at once. Then the same with a staggered arrival for comparison.

**Done when** it is known whether a class can start together, and if not, what pacing the runner has to impose.

### C. Retune the tier ladder

Your §2 table uses power-of-two catalogue tiers, but §5.E measured *tightest passing* slices, and those pack materially better. Applying `debit = nominal + 64` against the 16120 MiB pool to your own tight-slice numbers:

| workload | tightest passing | debit | fits per card | §2 said |
|---|---|---|---|---|
| resnet18 | 1728 | 1792 | **8** | 7 |
| bert | 3712 | 3776 | **4** | 3 |
| sd | 4608 | 4672 | 3 | 3 |
| resnet50 | 6720 | 6784 | 2 | 2 |
| gpt2 | 6784 | 6848 | 2 | — |

Round numbers cost a whole slice at both the small and mid sizes. Derive a tier ladder chosen so that `N × (tier + 64) ≤ pool` for the N you actually want, rather than so the tiers look tidy, and confirm each tier empirically holds its intended workload class.

While there: **the 256 MiB pool reserve is worth justifying or shrinking.** At the 8-slice boundary it is the difference between 8 and 9 tenants — nine ResNet-18 slices need 16128 MiB against a 16120 pool, missing by 8 MiB. Establish what the reserve is protecting against and whether measurement supports its size.

### D. What a non-root tenant can actually do

Your bypass finding is the most consequential security result in the report: `env -u LD_PRELOAD` defeats the ceiling entirely, and `/etc/ld.so.preload` only holds if the tenant is not root and cannot supply a private loader or static binary.

So the platform will run tenants non-root — which is a decision with a user-experience cost nobody has measured. Characterize it: inside a non-root container with a read-only rootfs, what happens to `pip install`, `pip install --user`, `apt install`, conda, writing to `/tmp` and to a mounted work directory, building a CUDA extension, and running Jupyter or TensorBoard.

**Done when** there is a plain list of what a non-root tenant can and cannot do, so the platform can decide between pre-baked images, a writable overlay, or accepting some risk in a trusted campus population.

### E. Which constants are card-specific

Only one GPU was tested. The fleet is heterogeneous — mostly Ampere consumer cards, some older. The 290 MiB context, the 256 MiB internal charge, and the +32 MiB overshoot are all plausibly driver- and architecture-dependent, which makes the tier table 4080-Super-specific rather than universal.

You cannot test another card on this machine. What you can produce is a **short, self-contained calibration routine** that derives those three constants on any card in under ten minutes, plus a statement of which conclusions in the first report are card-specific and which are structural. Someone will run it on a 3090 later.

### F. Late-peak detection

Your §5.B concluded peaks arrive within seconds, so a too-small slice fails fast. Your own §6 contradicts this: GPT-2 overflowed during **evaluation**, late in the run, against a fragmented cache. Checkpointing has the same shape.

Establish how common late peaks are across the five workloads when evaluation and checkpointing are included in the measured window, and whether the runner's watcher can flag a job approaching its ceiling before it dies. A job that fails at 90 % completion wastes far more than one that fails at 2 %, and the runner already samples at 100 ms.

**Done when** the fail-fast claim is either qualified or replaced, and the runner can distinguish "safely inside its slice" from "about to hit the wall."

---

## 3. Two corrections to carry forward

**The SM limiter's non-work-conserving behavior is not purely a defect.** §7 item 10 recommends using it only as a priority lever, on the grounds that a capped job keeps sleeping on an idle card. That is right for research sharing. It is wrong for the exam case: identical enforced throughput regardless of how many classmates are running is precisely the fairness property the platform is meant to provide for graded work. Treat it as correct-for-one-use-case rather than as a flaw, and note both readings.

**Sharing buys capacity, not speed, and that was always the point.** Your makespan figures (0.996× to 1.15× versus back-to-back) are the honest result and should stay stated that way. The platform's case never rested on throughput — it rests on forty students being able to run *at all* on a dozen cards. Do not soften the number.

---

## 4. Out of scope, still

Everything excluded from the first handout remains excluded: REST API, web UI, authentication, multi-node, scheduling policy, credits and tiers as *administrative* decisions, performance prediction, storage sync, SSH routing, Kubernetes.

Also do not redo anything the first report settled. Enforcement, single-tenant correctness, pairwise co-tenancy, failure modes, and the runner's basic operation are established. This round is about scale, packing, and the operational consequences of the constraints you found.

---

## 5. Reporting

Same discipline as last time. A negative result is a successful outcome — if eight-way sharing degrades badly, that changes tier design and possibly the education argument, and the team needs it now rather than in April.

Two numbers will be used directly in platform decisions and should be stated plainly wherever they land: **the largest N at which per-job slowdown still tracks 1/N**, and **the tier ladder that follows from it**. Everything else supports those.
