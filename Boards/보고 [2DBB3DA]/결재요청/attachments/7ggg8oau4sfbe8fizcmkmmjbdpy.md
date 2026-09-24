# HAMi Verification — Second Round: Findings Report (task_handout_2)

**To:** the author of `task_handout_2.md`. Self-contained given that handout plus the first-round report (`findings_report.md`, whose §4 setup and definitions are assumed: host NVML v2 sampling at 100 ms, "host peak" includes the 290 MiB context, correctness = SHA-256 over the final `state_dict`, workload configurations in its §4.6).
**Status:** complete, 2026-09-02 (run 21:19–22:48 UTC, unattended, as user unit `tfx-round2`; every step succeeded). Same hardware and software as round 1 (RTX 4080 SUPER 16 GB, driver 580.178.04, HAMi-core `f01e9f23`, PyTorch 2.13.0+cu130). Nothing settled in round 1 was re-run.
**Reading guide:** §1 verdict, §2 the two headline numbers, §3–§8 tasks A–F (each with *Design* then *Result*), §9 the two corrections carried forward, §10 an erratum to report 1, §11 deviations, appendices.

---

## 1. Verdict

**Eight-way sharing does not degrade.** With N identical ResNet-18 tenants in tight 1728 MiB slices, per-job slowdown tracks 1/N to within 4 % at every N tested (2, 4, 6, 8, 9), aggregate throughput stays at 97–99 % of a solo job, the spread between the luckiest and unluckiest tenant never exceeds 5 %, every job is bitwise-correct, and there is no Xid. The limit is memory, not scheduling: nine tenants fit this card, a tenth gets a physical out-of-memory at start-up and the other nine are unaffected. Eight containers started at the same instant all reach training within 13 s with no initialisation failure, faster than staggering them. A tier ladder derived from measured tight slices and the real pool (15 904 MiB, not the 16 120 MiB assumed in round 1) packs 8 / 4 / 3 / 2 tenants and was confirmed at full packing with bitwise-correct results. Non-root tenants with a read-only rootfs can `pip install --user`, run Jupyter and TensorBoard and keep the ceiling un-bypassable, **provided their home is on an exec-capable volume** (a default `noexec` tmpfs breaks every pip-installed tool with a compiled extension). Late peaks are rarer than round 1 feared: no evaluation or checkpoint phase allocated more than training in any workload; the one late failure is a phase that needs a single block larger than anything before it, which no watcher can predict but a 95 %-of-slice warning does flag two minutes ahead.

## 2. The two numbers for platform decisions

| number | value |
|---|---|
| **Largest N at which per-job slowdown still tracks 1/N** | **9** on this card — every N from 2 to 9 met the rule (Σ throughput ≥ 0.90 × solo and max slowdown ≤ 1.15 N, all bitwise-correct). N is bounded by memory (10 × 1760 MiB > 15 904), not by time-slicing. Efficiency at N = 8: 0.993; fairness spread 4 %. |
| **Tier ladder** (rule: N × (tier + 64) ≤ 15 904) | **N = 8 → 1920 MiB, N = 6 → 2560, N = 4 → 3904, N = 3 → 5184, N = 2 → 7872, N = 1 → 15 808** (usable inside = tier − 256). Confirmed at full packing: 4 × BERT @3904, 3 × SD @5184, ResNet-50 + GPT-2 @7872, all correct. |

---

## 3. Task A — high-N concurrency (the blocker)

**Design.** Workload: ResNet-18/CIFAR-10 exactly as in round 1 (3 epochs, batch 128, 4 dataloader workers), each in its own 1728 MiB slice (the round-1 tightest passing slice). Configurations, launched 1 s apart from one host process under one sampler: N = 1 (a fresh solo in the same session, reference), 2, 4, 6, 8, 9, and 10. Nine tenants fit the physical card only at the measured +32 MiB overshoot (9 × 1760 = 15 840 ≤ 15 904); ten is a deliberate physical overcommit. Per job: container wall time, training throughput, SHA vs the round-1 baseline. Per configuration: per-job slowdown = wall / solo wall; aggregate throughput = Σ (job throughput / solo throughput); fairness = max slowdown / min slowdown; makespan; host CPU utilisation (8 cores) from the sampler. CPU control: the host has 8 cores and each container runs 4 loader workers, so N = 8 means 40 processes; a second solo and N = 8 with `--workers 1` isolates GPU time-slicing from CPU contention. Two mixed configurations: ResNet-50 @6976 + 5 × ResNet-18 (N = 6, 15 744 MiB debited) and BERT @3968 + 6 × ResNet-18 (N = 7). Rule for "tracks 1/N": aggregate ≥ 0.90 and max slowdown ≤ 1.15 N with all jobs correct.

**Result — uniform tenants.**
| N | per-job slowdown min / max / mean | ideal | Σ throughput / solo | fairness max/min | all bitwise-correct | makespan | host CPU mean / p95 | resident VRAM |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.00 (42.6 s, 3835 img/s) | 1 | 1.000 | — | yes | 42.6 s | 26 % | 1.7 GB |
| 2 | 2.00 / 2.01 / 2.00 | 2 | 0.971 | 1.002 | yes | 86 s | 40 / 63 % | 3.4 GB |
| 4 | 3.86 / 3.92 / 3.90 | 4 | 0.978 | 1.016 | yes | 169 s | 68 / 99 % | 6.7 GB |
| 6 | 5.67 / 5.83 / 5.78 | 6 | 0.985 | 1.028 | yes | 252 s | 87 / 100 % | 10.1 GB |
| **8** | **7.45 / 7.75 / 7.67** | 8 | **0.993** | **1.040** | **yes** | 335 s | 99 / 100 % | 13.4 GB |
| 9 | 8.30 / 8.71 / 8.62 | 9 | 0.991 | 1.049 | yes | 376 s | 99 / 100 % | 15.1 GB |
| 10 (overcommit) | nine jobs 8.34–8.69; **tenth: physical CUDA OOM at 10 s** | 10 | 0.991 (nine) | — | nine yes, one failed | 376 s | 99 / 100 % | 15.9 GB (card full) |
| 8, `--workers 1` control | 7.10 / 7.42 / 7.34 (vs its own solo 44.5 s) | 8 | 1.042 | 1.045 | yes (all equal to the 1-worker solo's SHA) | 335 s | 99 / 100 % | 13.4 GB |

- **Slowdown tracks 1/N up to the physical memory limit.** Nothing between 4 and 8 collapses; efficiency actually improves slightly with N (0.97 → 0.99) because per-job idle gaps get filled. Time-slice fairness holds: the worst tenant is at most 5 % slower than the best.
- **CPU is not the limiter.** At N ≥ 6 the eight host cores sit at 99 %, but aggregate throughput stays at 0.99; with one loader worker per container (CPU comfortably under the limit) the curve is the same (7.34 × at N = 8, aggregate 1.04). The 1-worker jobs hash differently from the 4-worker baseline only because augmentation random streams are partitioned per worker; all eight matched their own 1-worker solo.
- **Overcommit fails cleanly and hits the newcomer.** The tenth container's first allocations found the physical card full ("GPU 0 has a total capacity of 1.69 GiB of which 968 MiB is free", eight other processes listed) and it exited with a CUDA OOM at 10 s; the nine resident tenants completed bitwise-correct. This is the hardware-level failure mode a mis-sized pool produces: fast and confined to the tenant that arrived last, *because* every tenant here allocates its peak at start-up. A late-allocating tenant would have been the victim instead, which is why the pool must never overcommit.
- **Practical reading for coursework:** eight ResNet-18-class jobs on one card each take 335 s instead of 43 s. Capacity, not speed, as established in round 1.

**Result — mixed tenants.**
| configuration (N) | large tenant slowdown | small tenants slowdown | correct | makespan | resident VRAM |
|---|---|---|---|---|---|
| ResNet-50 @6976 + 5 × ResNet-18 @1728 (6) | **3.31 ×** (97 img/s vs 330) | 5.85–5.97 × | all | 296 s | 15.1 GB |
| BERT @3968 + 6 × ResNet-18 @1728 (7) | 6.39 × | 6.41–6.65 × | all | 288 s | 13.7 GB |

The time-slicer is fair per *kernel time*, not per tenant: ResNet-50's long kernels win it roughly twice the share of a ResNet-18 (3.3 × instead of 6 ×), while BERT's kernel mix shares evenly with the small tenants. A real lab session with one heavy job will therefore see the heavy job favoured, consistent with the round-1 long-kernel adversary. Correctness holds in both mixes; no Xid anywhere in task A.

## 4. Task B — startup storm

**Design.** Eight ResNet-18 containers in 1728 MiB slices launched from eight host threads with zero stagger (Docker API calls issued concurrently), then the same with a 2 s stagger. Per container, from the launch instant: Docker `StartedAt`, first GPU memory seen by the host sampler (CUDA context created), and first 20 training steps completed (from the workload's timeline). Then pool accounting under a submission storm: 16 jobs submitted to the runner from 16 threads at once with `max_concurrent = 16`, so admission is limited only by memory; the runner's status was polled every 250 ms for the number running and the pool's debited total.

**Result.**
| storm | init failures | Docker start | CUDA context up | first tenant training | **all eight training** | correct | CPU mean / p95 |
|---|---|---|---|---|---|---|---|
| 8 simultaneous | 0 | 0.67–0.75 s | 3.4–3.7 s (all eight) | 8.4 s | **12.8 s** | all | 99 / 100 % |
| 8 with 2 s stagger | 0 | 0.6–14.6 s | 3.0–17.6 s | 4.4 s | 28.2 s | all | 98 / 100 % |

- **A class can start together.** Eight simultaneous CUDA initialisations complete within 0.3 s of each other at ~3.5 s; none fail. Reaching the first training steps takes 8–13 s, dominated by 32 dataloader workers competing for 8 cores, not by the GPU.
- Staggering buys nothing: each container's own initialisation takes the same ~3 s, so a 2 s stagger simply adds 14 s to the moment the last one is training. **The runner does not need to impose pacing for eight.** (Forty clicks would still be queued by memory, below.)
- **Pool accounting under a storm is correct:** 16 simultaneous submissions → exactly 8 running at once (memory allows 8 × 1792 within the 15 840 MiB capacity), zero samples with the debited total above capacity, 16/16 completed, capacity back to 15 840/15 840, no leaked processes; 677 s wall for the two waves.

## 5. Task C — tier ladder and the pool reserve

**Design.** Pool = NVML *free* memory at idle (v2 struct: total 16 376 − driver-reserved 455 − 17 used = **15 904 MiB**). Debit per slice = nominal + 32 (measured overshoot) + 32 (margin) = nominal + 64. Ladder rule: tier(N) = largest 64-aligned value with N × (tier + 64) ≤ 15 904, for N = 8, 6, 4, 3, 2, 1. Confirmation at full packing, each tenant in its tier, launched 1 s apart: 4 × BERT @T4, 3 × Stable Diffusion @T3, ResNet-50 + GPT-2 @T2. The reserve question is answered by the N = 9 / N = 10 runs of task A and by what NVML already excludes.

**Result — the ladder.**
| N per card | tier (nominal) | usable inside (tier − 256) | N × debit | spare | measured workloads that fit (tight slice) |
|---|---|---|---|---|---|
| 8 | **1920** | 1664 | 15 872 | 32 | ResNet-18 (1728) |
| 6 | 2560 | 2304 | 15 744 | 160 | ResNet-18 |
| 4 | **3904** | 3648 | 15 872 | 32 | BERT (3712) |
| 3 | **5184** | 4928 | 15 744 | 160 | SD fp16 (4608) |
| 2 | **7872** | 7616 | 15 872 | 32 | ResNet-50 (6720), GPT-2 (6784) |
| 1 | 15 808 | 15 552 | 15 872 | 32 | everything measured |

Compared with round 1's power-of-two catalogue: ResNet-18 8 per card instead of 7, BERT 4 instead of 3, ResNet-50/GPT-2 2 instead of 1. The handout's own table (8 / 4 / 3 / 2 / 2) is confirmed exactly.

**Result — confirmation at full packing** (all tenants bitwise- or PPL-correct, 0 Xid):
| packing | resident VRAM at peak | physical headroom left | makespan (vs solo) |
|---|---|---|---|
| 4 × BERT @3904 | 14 657 MiB | 1247 MiB | 158 s (solo 41 s → 3.85 × each) |
| 3 × SD @5184 | 13 764 MiB | 2140 MiB | 99 s |
| ResNet-50 + GPT-2 @7872 | 13 425 MiB | 2480 MiB | 224 s |

**Result — the reserve.** Round 1's 256 MiB reserve was applied to the wrong base: the pool was taken as total − 256 = 16 120 MiB, but the card can only ever hand out its NVML *free* figure, 15 904 MiB, because the driver holds 455 MiB itself. Round 1 therefore overcommitted by 216 MiB and was saved by the 2× overshoot margin. What a reserve actually protects against, measured: (a) the driver's own reservation, already excluded from "free"; (b) the +32 MiB per slice, already in the debit; (c) rounding. Nine ResNet-18 tenants at 1728 (9 × 1760 = 15 840) ran correctly with 64 MiB to spare card-wide; ten did not fit. **A 64 MiB card-wide reserve on top of NVML free is sufficient and supported by measurement; 256 MiB is not needed.** With that reserve the ninth 1728 MiB slice (9 × 1792 = 16 128 > 15 840) is refused by the +64 debit rule while physically fitting at +32; if the ninth seat matters, the debit margin can be reduced from 64 to 32 (the measured overshoot is a constant), which the N = 9 run supports. On a card with a display attached, NVML free already accounts for it.

## 6. Task D — what a non-root tenant can actually do

**Design.** One `tfx-workload` container as uid 1000 (no root), `--read-only` rootfs, `--tmpfs /tmp` (Docker default options: `rw,nosuid,nodev,noexec`, 6 GB), a writable bind-mounted work directory, `/etc/ld.so.preload` bind-mounted read-only with the HAMi library path, `--cap-drop ALL`, `--security-opt no-new-privileges`, network enabled (pip), 4 GB slice. A script runs each action and records exit status. Pass 1: HOME on the tmpfs. Pass 2: HOME on the writable bind mount (an ordinary ext4 volume, exec allowed) — what a platform-provided home volume would be.

**Result.**
| action | HOME on `noexec` tmpfs (Docker default) | HOME on exec-capable volume |
|---|---|---|
| write to /tmp, to the work mount | works | works |
| write anywhere in the rootfs | refused (read-only) | same |
| `pip install <pkg>` (system) | pip silently falls back to `--user` (site-packages not writable) | same |
| `pip install --user <pkg>`, `import` | works (pure-Python packages) | works |
| run a pip-installed command (`~/.local/bin/x`) | **fails: Permission denied (noexec)**; `python -m x` works for pure-Python tools | works |
| `pip install --user jupyterlab`, start Jupyter Lab | install ok; **start fails even via `python -m`: compiled extension (pyzmq) "failed to map segment" from noexec tmpfs** | **works, listening in < 90 s** |
| `pip install --user tensorboard`, start TensorBoard | install ok; **start fails (grpc extension, same cause)** | **works** |
| `apt-get install` | refused (not root, dpkg lock) | same |
| conda | not present in the image | same |
| build a CUDA extension | impossible: no `nvcc` in the runtime image (C++-only path not validly tested, see §11) | same |
| CUDA allocation inside the slice | works, sees 4096 MiB | same |
| `env -u LD_PRELOAD` then allocate 6 GiB | **refused** (ceiling holds via /etc/ld.so.preload) | same |
| write /etc/ld.so.preload | refused (read-only) | same |
| private copy of the glibc loader, run python through it, allocate 6 GiB | **refused** — the copied loader still honours /etc/ld.so.preload (copy on tmpfs cannot even execute) | same |
| Hugging Face cache on the mounted data volume | writable | same |
| disk used by a Jupyter + TensorBoard user install | ~200 MB | ~200 MB |

Plain list for the platform decision:
- **Can:** install pure-Python and wheel-based packages per user, run Jupyter Lab and TensorBoard, use the shared model/dataset cache, allocate GPU memory up to the slice — all as non-root on a read-only image — *if* HOME lives on an exec-capable writable volume. The pre-baked image + per-user home volume model works without any privilege.
- **Cannot:** install system packages (apt), modify the image, build CUDA extensions (needs a devel image with nvcc), escape the ceiling by any of the three tested routes.
- **Trap to design out:** Docker's default tmpfs is `noexec`; using it as HOME makes every pip-installed tool with a compiled extension unusable, with a confusing "failed to map segment" error. Mount the home volume `exec`, or bind a real directory.
- Also observed: a container carrying `/etc/ld.so.preload` cannot start *any* process unless the GPU libraries are injected (the preload needs `libcuda.so.1`); CPU-only containers must not carry the preload file.

## 7. Task E — which constants are card-specific

**Design.** `src/calibrate/calibrate_card.py`: on any card with Docker + the toolkit + the workload image + `bin/libvgpu.so`, it (1) reads NVML total/used/free/reserved at idle, (2) runs a context-only process natively and under a 4 GiB slice to get the physical context cost and HAMi's in-slice charge, (3) runs the step allocator under the slice to get usable and overshoot, (4) runs it natively to get the physical ceiling, and prints the three constants plus the derived pool, debit rule and the largest tier for N = 1…10. Runtime here: 28 s. Output `results/calibration_<card>.json`.

**Result on this card** (reproduces round 1 exactly): C_ctx = 290 MiB, HAMi context charge = 250 MiB (usable loss 256 with rounding), overshoot = 32 MiB, pool = 15 904 MiB, tier for N = 8 = 1920 MiB.

**Card-specific vs structural** (what a 3090 run of the routine would and would not change):
| claim | status |
|---|---|
| 290 MiB physical context; 250 MiB HAMi charge; +32 overshoot; 455 MiB driver reserve | **card- and driver-specific.** Context size depends on architecture, driver, CUDA runtime and the libraries loaded; HAMi's 250 is a constant in its source but the real context is not. Re-derive per card and per image. |
| usable = nominal − (charge + rounding); footprint = nominal + (C_ctx − charge); debit = footprint + margin; pool = NVML free | **structural** (follows from HAMi's accounting design); only the numbers move. |
| ceiling is real; overrun is a catchable CUDA OOM; per-slice binding; NVML virtualised in-slice | structural (userspace interception, no hardware dependence) |
| slowdown ≈ 1/N, fairness within 5 %, aggregate ≈ 0.99, storm start-up ~3.5 s | **partly card-specific.** Time-slicing is a driver/GPU feature (Pascal+ compute preemption); the *shape* should hold on Ampere, the absolute throughputs and context-switch overhead may differ. Re-measure N = 8 on the target card with the same driver. |
| tight-slice sizes per workload (1728 / 3712 / 4608 / 6720 / 6784) | card-specific through the context term (peak = allocations + C_ctx) and cuDNN workspace choices; re-measure or adjust by ΔC_ctx. |
| SM-limit semantics (not a share of compute; not work-conserving) | structural to HAMi's token bucket; magnitudes workload- and card-dependent. |
| bypass via env, fix via /etc/ld.so.preload; noexec-tmpfs trap | structural (Linux/Docker), not GPU-related. |
| tier ladder 1920 / 2560 / 3904 / 5184 / 7872 | **this card only**; recompute from the calibration output. |

## 8. Task F — late peaks and early warning

**Design.** F1: each workload confined at its tight slice + 256 MiB with per-epoch checkpointing enabled (model + optimizer via `torch.save` to the results volume) and evaluation inside the window; the workload records per-phase peaks of *allocated* (live tensors) and *reserved* memory with phase labels train / eval / ckpt (SD: load / generate); the host sampler gives the physical figure and when it peaked. F2: GPT-2 with the original evaluation batch of 16 and no cache flush, in a slice sized for its training (7040 MiB): the round-1 late failure, reproduced under confinement. F3: the runner's watcher now estimates the tenant's HAMi-side usage as host_used − 40 MiB, emits WARN at 90 % and CRITICAL at 97 % of nominal, with an ETA from the growth rate over the last 20 samples; validated with a probe that grows by 64 MiB every 0.5 s (fast) and 32 MiB every 3 s (slow) in a 2048 MiB slice, plus the GPT-2 eval-16 job and a healthy ResNet-18 in its 1984 tier.

**Result — where peaks really are.**
| workload | live-tensor peak by phase (MiB) train / eval / ckpt | physical process footprint by phase (host) | t90 | checkpoint size |
|---|---|---|---|---|
| ResNet-18 @1984 | 963 / 454 / 194 | 1672 in all phases | 0.6 s | 85 MB × 3 |
| ResNet-50 @6976 | 5614 / 1017 / 342 | 6662 in all phases | 1.2 s | 180 MB × 3 |
| BERT @3968 | 2974 / 2017 / 1753 | 3654 in all phases | 4.4 s | 1253 MB |
| GPT-2 @7040 | 5523 / 3985 / 2514 | 6730 train, 6336 eval, 5760 ckpt | 12.7 s | 1424 MB |
| SD @4864 | load 2057 / generate 3114 | 2772 load, 4576 generate | 5.8 s | — |

- **Evaluation and checkpointing never exceeded training** in any of the five workloads: their live allocations are 40–80 % *lower*, and checkpointing (a device-to-host copy) adds no VRAM at all. The physical footprint reaches 90 % within 0.6–12.7 s and then stays flat because PyTorch's cache is never returned. Round 1's fail-fast claim stands **for phases whose largest single allocation is no bigger than training's**.
- **The qualification:** a phase that needs one block larger than anything training requested fails late. GPT-2 with evaluation batch 16 needs a 1.54 GiB logits block; in the 7040 MiB slice it trained all 1100 steps and died in the final evaluation at **94 % of the run** (135 s in), a clean CUDA OOM. Same shape as an evaluation with a larger batch than training, or a generation step with a long sequence. Checkpointing does not have this shape.
- **Rule that follows:** size a slice from a measured peak that *includes the job's evaluation batch*, and treat the evaluation batch as part of the job's memory spec. The simplest guard in user code is the one that fixed GPT-2 in round 1: evaluate with a batch no larger than training, and `torch.cuda.empty_cache()` before a phase change.

**Result — the watcher.**
| job | slice | warnings emitted (fraction of nominal, headroom, ETA) | actual OOM | lead time |
|---|---|---|---|---|
| grow 128 MiB/s | 2048 | WARN at 91 % (176 MiB left, ETA 1.3 s); CRITICAL at 98 % (48 MiB, ETA 0.4 s) | 13.7 s | 1.5 s / 0.5 s |
| grow 10.7 MiB/s | 2048 | WARN at 91 % (ETA 13.1 s); CRITICAL at 98 % (ETA 3.6 s) | 76.2 s | **14.9 s / 2.8 s** |
| GPT-2 eval-16 | 7040 | WARN at 96 % from t = 11 s (training already at 6800 MiB); no CRITICAL | at 135 s (eval) | 124 s of "no headroom", but no prediction of the jump |
| ResNet-18 in its tier | 1984 | none (82 % of nominal) | — | — |

- Gradual growth is caught: lead time ≈ headroom / growth rate and the ETA estimate is within 15 % of the true time to the wall, so a job leaking or accumulating memory gets seconds to minutes of notice.
- Sudden phase-change allocations are not predictable from the sampler: the GPT-2 request appeared and failed inside one 100 ms interval. What the watcher *does* provide for that case is the persistent WARN: a job running at 96 % of its slice for two minutes has no room for any phase change. **The runner can now distinguish "safely inside its slice" (no warning; ResNet-18 at 82 %) from "about to hit the wall" (WARN/CRITICAL with ETA, or a persistent > 95 % state).** Warnings are recorded in the job record and the ledger (`warning:WARN`, `warning:CRITICAL` events) for a UI to surface.

## 9. Two corrections carried forward (handout §3)

- **SM limiter, non-work-conserving.** Round 1 §7 item 10 called the capped job "sleeping on an idle card" a flaw. Both readings are now stated: for *research sharing* it wastes capacity and should be applied only under contention; for *graded coursework* it is the desired property — every student gets the same enforced throughput regardless of how many classmates are running, which is exactly what makes results comparable. The platform should expose it as "fixed-rate seat" for graded sessions and "priority" for research, not as a percentage of performance (its non-proportionality from round 1 §5.S still holds).
- **Sharing buys capacity, not speed.** The round-1 makespan figures (0.996–1.15 × versus back-to-back) stand unchanged, and this round adds the N-way version: eight tenants each run at ~1/8 speed with 99 % of the card's throughput preserved. The platform's case rests on forty students running at all on a dozen cards; this round shows that eight per card do, correctly and fairly.

## 10. Erratum to the first report

Report 1 §2 ("slices per card") used a pool of 16 120 MiB (total − 256). The card's allocatable memory is its NVML *free* figure, 15 904 MiB (driver reserves 455 MiB). The corrected pool with a 64 MiB reserve is 15 840 MiB. The counts in that table (7 / 5 / 3 / 3 / 2 / 1 for the power-of-two tiers) happen to be unchanged, but the ladder in §5 above supersedes the table, and the runner's `GpuPool` now uses NVML free at start-up. A note has been added to report 1.

## 11. Deviations and limits

- **CPU saturation at N ≥ 6** (99 % of 8 cores from 32 dataloader workers) is a property of this small host, not of the GPU sharing; the 1-worker control shows the GPU curve is unaffected. A lab server with more cores per GPU would not see it; one with fewer would, and the symptom is slower *start-up* (task B), not lower aggregate throughput.
- **"Ideal" for the mixed configurations is not 1/N**: the time-slicer shares kernel time, so a long-kernel tenant legitimately gets more. Reported as measured rather than forced into the 1/N rule.
- **Task D's C++-extension test was invalid** (the test passed a non-existent build directory); CUDA extensions are impossible in the runtime image anyway (no `nvcc`). A devel-image variant was not tested.
- **The `--workers 1` jobs "fail" the SHA check against the 4-worker baseline** by construction (augmentation RNG is partitioned per worker); they were checked against their own 1-worker solo instead and all eight matched.
- **Smoke run.** A reduced-size pass of every task ran first; it exposed three test defects (phase peaks measured on reserved memory, which never shrinks; a package already in the image; and the noexec-tmpfs interaction) that were fixed before the full run reached those steps. Smoke outputs are kept separately in `results/smoke/` and are not used anywhere.
- The whole round ran unattended (user unit `tfx-round2`) after the smoke; GPU time ≈ 1.5 h.

## Appendix A — files
`results/r2a_summary.json` (task A, plus `results/r2a_*/` per-configuration directories with per-job `run_meta.json`, sampler CSV and workload JSON), `r2b_summary.json` and `results/runner_r2b/` (ledger of the 16-job storm), `r2c_summary.json`, `r2f_summary.json` and `results/runner_r2f/`, `results/r2d_nonroot/nonroot_ux.json` + `container.log`, `results/calibration_NVIDIA_GeForce_RTX_4080_SUPER.json`, `results/round2_summary.json` (everything merged), auto-generated tables in `documentations/findings_report_2_draft.md`. Drivers: `src/experiments/round2_{a,b,c,d,f}.py`, `nonroot_ux.sh`, `round2_analyze.py`; `src/calibrate/calibrate_card.py`; probe `src/probe/grow_alloc.py`; runner changes in `src/runner/tfx_runner/{pool,watcher,runner,models}.py`. Running log: `.agent_logs/2026-09-02_task2_digest_and_plan.md`.

## Appendix B — reproduction
```
bin/run_round2.sh --smoke        # ~8 min, tiny sizes, outputs to be discarded
bin/run_round2.sh                # ~1.5 h: A → C → B → F → D → E → analyze (each step resumable, results skipped if present)
venv/bin/python src/calibrate/calibrate_card.py          # any card, < 1 min here
venv/bin/python src/experiments/round2_a.py              # a single task
```
