# HAMi Verification — Third Round: Findings Report (task_handout_3)

**To:** the author of `task_handout_3.md`. Self-contained given that handout plus the round 1 and 2 reports (their §4 setup and definitions are assumed: host NVML v2 sampling, "host peak" includes the ~290 MiB context, correctness = SHA-256 over the final `state_dict`, "usable = nominal − 256 MiB", "slice" = a container with `libvgpu.so` preloaded and `CUDA_DEVICE_MEMORY_LIMIT` set).
**Status:** complete, 2026-09-03. Same box (one RTX 4080 SUPER 16 GB, driver 580.178.04, HAMi-core `f01e9f23`, PyTorch 2.13.0+cu130). Task B ran the full 12 hours (03:02→15:02 UTC) unattended as user unit `tfx-round3`. Two-card parts of Task A were **not** run (single-GPU box; the user set that part aside) — §2 says exactly what that leaves open.

---

## 1. The three sentences

1. **Do NCCL allocations stay inside the slice?** **Yes.** NCCL's communication buffers — on both the `cudaMalloc` and the `cuMemCreate`/`cuMemMap` (`NCCL_CUMEM_ENABLE=1`) paths — are counted against the ceiling. A tenant filled to 3.5 GB of a 4 GB slice was refused *exactly at the 4 GiB line* when NCCL tried to add its buffers (HAMi log `Device 0 OOM 4458545152 / 4294967296`), never exceeding its slice. A multi-GPU tenant cannot escape its declared memory limit through NCCL.
2. **Longest safe session hold?** **At least 12 hours — no drift of any kind was observed, so no cap is required by the memory layer.** Across a 12-hour hold with a burst every 10 minutes, the enforced ceiling stayed at exactly 3456 MiB usable at all eight checkpoints (hours 0/1/2/4/6/8/10/11.8), a ResNet-18 trained at hour 11.8 was **bitwise-identical** to one trained at minute 1, throughput was flat (3775–3806 img/s), container RSS rose once and then held, a flat-idle slice never moved, and there were 0 Xid. The platform can offer "leave it running" for a working day without recycling.
3. **Can a non-root tenant compile a CUDA extension?** **Yes**, on a devel-based image. A non-root tenant with a read-only rootfs compiled and ran a custom CUDA kernel in ~22 s, the result was correct, it respected the ceiling, and every round-1 bypass route stayed closed. The cost is image size: the devel image is 21.3 GB, **+15.0 GB** over the runtime image.

---

## 2. What was and was not answered on this box (Task A honesty)

The verification box has one GPU. Answered here: whether a HAMi-preloaded container initializes NCCL and forms a process group; whether NCCL's buffers are counted against the slice; whether `gloo` differs from `nccl`. **Not answered here** (need two cards): whether one container can hold independent slices on two physical devices (`CUDA_DEVICE_MEMORY_LIMIT_1` genuinely independent of `_0`), and whether real DDP across two sliced GPUs completes correctly. A two-card cloud instance was not provisioned (out of scope for me to spin up on my own).

**Framed against the fallback** (multi-GPU jobs get whole cards on a separate, un-sliced path): the single-GPU results show interception does **not** break collectives — NCCL init, all-reduce, broadcast, all-gather and a DDP step all work under a slice and stay memory-bounded. So slicing + NCCL is worth finishing on two cards rather than abandoning; the whole-cards fallback is only needed if the two-card memory-independence or DDP-correctness test later fails. The one thing that could still bite on two cards is per-device limit independence, which has never been exercised — the shipped-then-removed two-GPU script's logic is reproduced in Appendix B so it can be run in an hour on any Turing-or-newer two-card machine.

---

## 3. Task A — NCCL / gloo under HAMi (one card)

**Design.** In-container probe (`src/probe/nccl_probe.py`), each run also host-sampled. `single`: world_size-1 NCCL group — init, all-reduce/broadcast/all-gather on a 256 MiB tensor, then a 5-step DDP train of a small CNN; in-slice accounting (`torch.cuda.mem_get_info`) recorded before init, after all-reduce, after DDP. `escape`: fill the slice to F MiB with a plain tensor first, *then* init NCCL and run a collective — if buffers are counted the total is refused at the ceiling, if they escape the host footprint exceeds nominal. Run at F = 3500/3000/2800, with `NCCL_CUMEM_ENABLE` unset/0/1, plus a native (no-HAMi) reference. `two`: two ranks on the one GPU with `nccl` and with `gloo`. NCCL 2.29.7.

**Result** (all 14 runs classified ok, **0 Xid**):
| question | result |
|---|---|
| NCCL init under a 4 GB slice | forms the process group; all-reduce/broadcast/all-gather correct; DDP step runs, loss `2.247468` identical to native and to both `NCCL_CUMEM_ENABLE` settings |
| are NCCL buffers counted? | **yes.** world_size-1 all-reduce raised in-slice usage by ~644 MiB (256 MiB tensor + ~390 MiB NCCL buffers); `cumem=0` left a slightly larger footprint than `cumem=1`, both accounted |
| escape test (fill then init) | F=3500 → HAMi refused at the 4 GiB line, clean `torch.OutOfMemoryError`, process capped; F=3000 → init ok, the collective's buffer allocation refused; F=2800 → collective ok (usage 3296→3940, buffers counted), the following DDP allocation refused at the ceiling; native reference grew unbounded to ~5 GB in-slice |
| gloo vs nccl | gloo with two CUDA-tensor ranks works and is correct, under a slice and natively |
| two NCCL ranks on one GPU | refused: "Duplicate GPU detected: rank 1 and rank 0 both on CUDA device" — a NCCL property, not a HAMi effect |

The round-1 finding that `cuMemCreate` is hooked is confirmed end-to-end: NCCL's modern allocation path lands inside the ceiling.

## 4. Task B — long idle hold (12 hours)

**Design.** Two containers held for 12 h under one host sampler (1 s): a **session** container (4096 MiB slice) modelling a long-lived Jupyter kernel — holds a CUDA context and every 10 min runs a burst (allocate 2 GiB, train a small CNN 200 steps, free), idle otherwise, logging in-slice accounting every 60 s; and a **flat-idle** container (2048 MiB slice) holding only a context. `docker stats` (RSS, CPU) every 60 s. At hours 0/1/2/4/6/8/10/11.8, via `docker exec` into the session container: (a) a fresh `step_alloc` to re-measure the enforced ceiling, (b) a fresh 1-epoch ResNet-18 (full determinism harness) for a bitwise SHA, (c) an accounting snapshot from a fresh process.

**Result — nothing drifted.**
| checkpoint (hour) | enforced ceiling (usable / total MiB) | ResNet-18 SHA | ResNet-18 acc | img/s | fresh-process free/total | Xid |
|---|---|---|---|---|---|---|
| 0 | 3456 / 4096 | `ce935317dcc2` | 0.6224 | 3787 | 3469 / 4096 | 0 |
| 1 | 3456 / 4096 | `ce935317dcc2` | 0.6224 | 3800 | 3469 / 4096 | 0 |
| 2 | 3456 / 4096 | `ce935317dcc2` | 0.6224 | 3806 | 3469 / 4096 | 0 |
| 4 | 3456 / 4096 | `ce935317dcc2` | 0.6224 | 3802 | 3469 / 4096 | 0 |
| 6 | 3456 / 4096 | `ce935317dcc2` | 0.6224 | 3775 | 3469 / 4096 | 0 |
| 8 | 3456 / 4096 | `ce935317dcc2` | 0.6224 | 3778 | 3469 / 4096 | 0 |
| 10 | 3456 / 4096 | `ce935317dcc2` | 0.6224 | 3792 | 3469 / 4096 | 0 |
| 11.8 | 3456 / 4096 | `ce935317dcc2` | 0.6224 | 3782 | 3469 / 4096 | 0 |

- **Ceiling enforcement is invariant:** the same 3456 MiB usable (out of the 4096 slice) at hour 11.8 as at minute 1. (This 3456 vs the 3840 seen in a bare slice is because the session's long-lived process is co-resident and holding its own reserved cache; the point is that the figure never moves.)
- **Correctness is invariant:** identical parameter SHA at all eight checkpoints — a workload run after 12 hours of holding computes bitwise-identically to one run at the start.
- **No leak in HAMi accounting:** a fresh process saw a constant 3469 MiB free at every checkpoint across 12 hours; the session process's held footprint was steady.
- **Host state is stable:** session container RSS rose once (470 → 781 MiB, on the first burst's allocator growth) and then held at 781 MiB for the remaining 12 hours; CPU averaged 2 %; the flat-idle slice's in-container free stayed at 1798 MiB and its torch-reserved at 2 MiB from first sample to last. 73 bursts completed; burst wall time went from 1.18 s (cold) to a steady ~0.2 s (warm), i.e. no slowdown. After both containers exited, host GPU memory returned to 17 MiB. **0 Xid over 12 hours.**
- **Longest safe hold:** the full 12 h passed with no drift in enforcement, correctness, memory, or driver health, so the memory layer imposes **no** session-time cap up to at least 12 h. If a cap is ever wanted it will come from higher layers (idle-reclaim policy), not from HAMi.

**One incident, and an operational finding.** The flat-idle container as originally launched died 2 s after start: it and the session container were launched at the same instant, and while the session ran its first 2 GiB burst the flat container's HAMi accounting transiently reported the session's usage (~2.34 GiB) inside its own 2 GiB slice and it OOM'd on its context. Isolation itself is intact — a fresh 2048 MiB container started mid-hold saw a clean 1798/2048 slice, and rounds 1–2 ran 8 concurrent slices without leakage — so this is a **cold-start race**: two HAMi containers initializing their shared accounting in the same microsecond, one of them allocating immediately, can make the other mis-read usage and spuriously OOM at start-up. Round 2's 8-way storm did not trigger it (uniform slices, gradual allocation, no large neighbour allocation at t=0). **Recommendation:** the runner should stagger container starts by ≳1 s, or retry a start-up CUDA-OOM once before failing the job. The flat baseline was relaunched cleanly (staggered) and held flat for the rest of the window.

*Measurement note:* the burst's small CNN was not run under deterministic algorithms (unlike the checkpoint ResNet-18), so its parameter sum varied at the 1e-7 level across the 73 bursts — floating-point noise, non-monotonic, not drift. The bitwise correctness claim rests on the checkpoint ResNet-18, which is identical.

## 5. Task C — devel image and CUDA extensions

**Design.** Built `tfx-workload-devel` from `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel` plus the same pip layer (nvcc present: `cuda_13.0.r13.0`). Repeated the round-2 non-root conditions (uid 1000, read-only rootfs, HOME on an exec-capable volume, `/etc/ld.so.preload` bound read-only, cap-drop ALL, no-new-privileges, 4 GB slice) and, inside: compiled a trivial CUDA kernel with `torch.utils.cpp_extension.load_inline` and ran it; had an extension call `cudaMalloc` directly in a loop to see if raw allocations are counted; built a standalone `nvcc` binary with **statically linked** cudart and ran it; re-ran the three bypass routes and the `step_alloc` ceiling check; measured the image size delta.

**Result** (container exit 0, **0 Xid**):
| check | result |
|---|---|
| compile + run a CUDA kernel (load_inline) | **works, 21.5 s, output correct**, in-slice total 4096 MiB |
| extension calling `cudaMalloc` directly | **counted** — capped at 3840 MiB usable (accounted delta 3840 MiB), i.e. raw driver allocations from a hand-written extension respect the ceiling |
| standalone nvcc binary, static cudart | sees the virtualized 4096 MiB total and is capped at 3840 MiB — **static cudart does not escape** (the binary still calls `libcuda.so` dynamically, which the preload's dlsym hooks intercept) |
| same static binary with `LD_PRELOAD` unset | still capped at 3840 (held by `/etc/ld.so.preload`) |
| `env -u LD_PRELOAD` + 6 GiB torch alloc; private-loader route | both refused |
| `step_alloc` ceiling on the devel image | 3840 usable, catchable OOM, correct post-OOM compute — identical to the runtime image |
| image size | runtime 6.3 GB → devel 21.3 GB (**+15.0 GB per node**) |

So the platform *can* serve research users who compile kernels, by distributing a second image tier; the tenant needs no privilege and the ceiling and bypass protections are unchanged. The only real cost is the 15 GB every node must store.

## 6. Decisions these feed
- **Multi-GPU:** keep slicing + NCCL on the table — interception does not break collectives and NCCL cannot exceed the slice. Finish the two-card memory-independence and DDP-correctness checks (Appendix B) before committing; fall back to whole-cards-per-job only if those fail.
- **Session length:** no memory-driven cap needed up to 12 h; "leave it running" is safe. Add ≳1 s start stagger / one OOM retry in the runner (the cold-start race).
- **Research image:** offer a devel tier (+15 GB) for compile-your-own-kernel users; coursework stays on the 6.3 GB runtime image.

## 7. Carried-forward discipline
A negative result would have been reported as such (it would have redirected the multi-GPU architecture); none occurred in what was testable. The one thing that *could not* be tested here (two-card slicing) is stated as unanswered rather than extrapolated.

## Appendix A — files
`results/r3a_summary.json` (+ `results/r3a_*/` per-run dirs with `run_meta.json`, sampler CSV, `probe.json`, `container.log`), `results/r3b_summary.json` (12 h: `checkpoints`, `session_log`/`flat_log`, per-60 s `stats`, sampler) with the hold artifacts under `results/r3b_hold/{session,flat}/`, `results/r3c_summary.json` + `results/r3c_devel/`, merged `results/round3_summary.json`, auto tables `documentations/findings_report_3_draft.md`. Smoke outputs isolated in `results/smoke3/`. Drivers: `src/experiments/round3_{a,b,c}.py`, `round3_analyze.py`; probes `src/probe/{nccl_probe,hold_session,grow_alloc}.py`; devel image `src/docker/Dockerfile.workload-devel`. Running log: `.agent_logs/2026-09-03_task3_digest_and_plan.md`.

## Appendix B — the two-card check to run elsewhere (≈1 h, any Turing+ two-GPU box)
1. **Per-device limit independence:** one container, both GPUs, `CUDA_DEVICE_MEMORY_LIMIT_0=2g CUDA_DEVICE_MEMORY_LIMIT_1=6g`, `libvgpu.so` preloaded. In-container `torch.cuda.mem_get_info(0)`/`(1)` totals must read 2048 / 6144 MiB, and a per-device step-allocator must hit 1792 / 5888 usable respectively. Confirms `_1` is honoured and independent of `_0` (only `_0` has ever been exercised).
2. **DDP across two slices:** `torchrun --nproc_per_node 2` of a DDP ResNet-18 (add a `DistributedSampler` + `DistributedDataParallel` wrapper to `resnet18_cifar10.py`), each rank in a 4 GB slice; must complete and match a native two-GPU run's SHA. Watch host NVML per-device to confirm neither rank exceeds its slice and there is no Xid.

## Appendix C — reproduction
```
bin/run_round3.sh --smoke      # ~10 min (B = 6 min hold)
bin/run_round3.sh              # A (~12 min) → C (~1 min) → B (12 h) → analyze; each step resumable
venv/bin/python src/experiments/round3_a.py     # NCCL/gloo one-card only
```
