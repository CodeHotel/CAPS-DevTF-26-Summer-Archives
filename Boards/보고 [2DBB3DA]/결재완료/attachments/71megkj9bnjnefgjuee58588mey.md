# HAMi-core Fractional-GPU Feasibility Verification — Findings Report

**To:** the agent that wrote `task_handout_1.md`. This report assumes you know only that handout. Every experimental condition is spelled out here; nothing refers to state outside this document except the raw files listed in Appendix C.
**Status:** complete, 2026-09-02. All phases A–H of the handout were executed, plus the SM-limit sweep the handout's synthetic-tool list implied. None of the six stop conditions fired.
**Reading guide:** §1 verdict, §2 the two design figures, §3 stop conditions. §4 is the full experimental setup and method; read it before the per-phase results in §5, because every number's meaning ("host peak", "usable", "nominal") is defined there. §6 deviations, §7 requirements, appendices for reproduction and raw data.

---

## 1. Verdict

**The assumption holds on this hardware and stack.** One RTX 4080 SUPER (16 GB) was split into HAMi-core slices whose VRAM ceilings are real (usable = nominal − 256 MiB, physical footprint = nominal + 32 MiB). Overrunning a ceiling produces a clean, catchable CUDA out-of-memory error with the driver untouched after more than a thousand consecutive OOMs. Five real workloads spanning CNN training, BERT and GPT-2 fine-tuning and Stable Diffusion inference produced **bitwise-identical results inside a slice, next to a hostile co-tenant, and across 204 runner jobs over a 3-hour soak**. Interception costs under 1 % throughput, confinement adds nothing measurable, and co-tenants cannot corrupt, kill or fault each other; they only slow each other down (about 2× for two saturating trainers, about 3× for short-kernel inference next to a trainer).

Three things the platform must build around: enforce the preload through `/etc/ld.so.preload` rather than an environment variable (the env route is trivially bypassed), size and account slices with the 256 MiB / +64 MiB rules in §2, and do not offer PyTorch's `cudaMallocAsync` allocator (CUDA-graph capture breaks under HAMi with it). The runner that ran the soak is in `src/runner/tfx_runner/` and is a usable base for the real system.

## 2. The two figures platform design needs

| figure | value | derived in |
|---|---|---|
| **Minimum headroom multiplier over measured peak** | **1.0× the host-measured native peak, rounded up to 64 MiB**, passed 3/3 for all five workloads. True floors are 5–25 % *below* the peak because PyTorch's caching allocator gives cache back under pressure. Recommended sizing rule: `slice_nominal = roundup64(peak_host) + 64 MiB`. | §5.E, §5.E3 |
| **Per-container fixed overhead** | **290 MiB physical VRAM per CUDA context** (identical with and without HAMi — HAMi itself adds 0 MiB of GPU memory, ~4 MiB RAM, ~1 % of a CPU core). In addition each slice loses **256 MiB of its nominal size** to HAMi's internal context charge, and its real footprint is **nominal + ~32 MiB**. | §5.D |

### Slices per card (the overhead figure applied N times)
Definitions: *nominal* = the value given to `CUDA_DEVICE_MEMORY_LIMIT`; *usable* = what the tenant's framework can actually allocate = nominal − 256 MiB; *physical debit* = what the card really loses = nominal + 64 MiB (measured +32, doubled for margin); *pool capacity* = 16376 − 256 MiB reserve = 16120 MiB.

| nominal | usable | physical debit | slices that fit on this card | measured workloads that fit inside |
|---|---|---|---|---|
| 2048 MiB | 1792 | 2112 | **7** | ResNet-18/CIFAR (peak 1672; 1258 when squeezed) |
| 3072 MiB | 2816 | 3136 | 5 | — |
| 4096 MiB | 3840 | 4160 | **3** (4 × 4160 = 16640 > 16120) | BERT-base/SST-2 (3654) |
| 4864 MiB | 4608 | 4928 | 3 | Stable Diffusion 1.5 fp16 (4576) |
| 7040 MiB | 6784 | 7104 | 2 | GPT-2 b4×512 (6730), ResNet-50 b64 @224 (6662) |
| 8192 MiB | 7936 | 8256 | **1** (2 × 8256 = 16512 > 16120) | — |

Four nominal 4 GB or two nominal 8 GB slices do **not** fit a 16 GB card. For the "forty students, a dozen cards" scenario, seven 2 GB slices per card covers ResNet-18-class coursework; BERT-sized work needs a 4 GB tier at three per card.

## 3. Stop conditions (handout §6)

| condition | result | evidence |
|---|---|---|
| Preload with no limit changes results or costs significant performance | **no** — identical results, +0.1…0.8 % wall (inside baseline noise) | §5.C |
| An allocation succeeds past the configured ceiling | **no** — bounded +32 MiB constant offset from context under-accounting, not a bypass; graph-captured and async allocations also refused | §5.D, §5.G |
| In-slice OOM → host fault / Xid / driver reset | **no** — 459 + 1096 consecutive OOMs, 0 Xid all day | §5.D, §5.F |
| Confined result materially differs from unconfined baseline | **no** — 20/20 confined runs bitwise (or PPL) identical | §5.E |
| Co-tenant can corrupt, kill, or fault another tenant | **no** — victim bitwise-correct under four adversaries, 0 Xid | §5.F |
| Ceiling binds per process rather than per slice | **no** — three processes in one slice share one budget | §5.F3 |

---

## 4. Experimental setup and method

### 4.1 Host
| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER, 16376 MiB, compute capability 8.9, VBIOS 95.03.44.00.D4, PCIe gen 3 link, 320 W limit; one GPU, no display attached (idle usage 17 MiB) |
| Driver | 580.178.04 (reports CUDA 13.0) |
| CPU / RAM / disk | Intel i7-9700KF, 8 cores (no SMT), 31 GiB RAM, 935 GB LVM root (227 GB free at end) |
| OS | Ubuntu 22.04.5 LTS, kernel 5.15.0-190-generic |
| Container runtime | Docker 29.1.3, overlay2, cgroup v2 (systemd driver); NVIDIA Container Toolkit 1.20.0 (libnvidia-container 1.20.0). No `nvidia` runtime registered in the daemon; `--gpus` works through the toolkit's prestart hook. |
| Host-side tooling | Python 3.10.12 venv with `nvidia-ml-py` (NVML bindings), `docker` SDK, numpy/pandas/pyyaml |
| Other load on the machine | none during timed runs. Four unrelated stopped containers existed and were left untouched. |

### 4.2 Software under test
- **HAMi-core** `github.com/Project-HAMi/HAMi-core`, commit **`f01e9f23fc6ab251d2a7fee8987279f16b08afc8`** (2026-08-31, "Merge PR #293 fix-nvml-v2-signatures"; the repository has no release tags). Pinned as a git submodule at `src/hami/HAMi-core`.
- Built exactly as upstream does: `make build-in-docker` → `nvidia/cuda:13.3.0-cudnn-devel-ubi8`, `build.sh` with `-DDLSYM_HOOK_ENABLE=1 -DMULTIPROCESS_LIMIT_ENABLE=1 -DHOOK_MEMINFO_ENABLE=1 -DHOOK_NVML_ENABLE=1 -DCMAKE_BUILD_TYPE=Debug` (Debug is what upstream's Dockerfile ships). Output `libvgpu.so`, 692 528 bytes, sha256 `9f930a1582baad71…`, copied to `bin/libvgpu.so`.
- **Authoritative environment variables** (from `grep getenv` over `src/`, not from the README):

| variable | meaning | notes |
|---|---|---|
| `CUDA_DEVICE_MEMORY_LIMIT` / `CUDA_DEVICE_MEMORY_LIMIT_<i>` | byte ceiling per device; suffix `g`/`m`/`k`, else bytes; per-device form overrides the plain one | verified: `4g`, `3072m`, `3221225472`, `_0=3g` all behave identically |
| `CUDA_DEVICE_SM_LIMIT` / `_<i>` | compute utilisation percent, default 100 = off | see §5.S |
| `CUDA_DEVICE_MEMORY_SHARED_CACHE` | path of the shared-memory accounting file, default `/tmp/cudevshr.cache` | each container has its own `/tmp`, so the budget is per container = per slice |
| `CONTAINER_VGPU_MOUNT`, `POD_UID`, `CONTAINER_NAME` | synthesise the cache path (Kubernetes use) | unused here |
| `ACTIVE_OOM_KILLER` | parsed, default on — but the kill function is **never called** at this commit (dead code) | OOM is always a returned error |
| `GPU_CORE_UTILIZATION_POLICY` | `FORCE` / `DISABLE` the SM limiter | not exercised |
| `CUDA_TASK_PRIORITY`, `RECORD_KERNEL_INTERVAL`, `LIBCUDA_LOG_LEVEL` (0–4), `CUDA_REDIRECT`, `CUDA_VISIBLE_DEVICES` | priority int, watcher interval, log verbosity, alt. libcuda path, device mapping | defaults used |
- README mentions creating `/tmp/vgpulock/`; that string does not exist in the source at this commit (stale documentation).
- Mechanism as read from source: `cuMemAlloc_v2`, `cuMemAllocManaged`, `cuMemAllocPitch_v2`, `cuMemAllocAsync`, `cuMemCreate` are hooked; `oom_check()` compares tracked usage + request against the limit and returns `CUDA_ERROR_OUT_OF_MEMORY`. NVML memory queries are hooked to report the slice. CUDA-graph *node* creation functions are passed through unhooked (relevant to §5.G). The SM limiter is a token bucket per device drained by kernel launches and refilled by a watcher thread; when empty the launching thread `nanosleep`s.

### 4.3 Workload container image (`tfx-workload`)
`FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime` (chosen to match the host driver's CUDA 13.0 and HAMi-core's own CUDA 13 build target), plus pip: transformers, datasets, diffusers, accelerate, safetensors, sentencepiece, scipy, pillow, scikit-learn, pandas, nvidia-ml-py. Exact versions inside the image:

| component | version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 (CUDA 13.0, cuDNN 9.20.0) |
| torchvision | 0.28.0+cu130 |
| transformers / datasets / diffusers / accelerate | 5.16.1 / 5.0.1 / 0.40.0 / 1.14.0 |
| triton | 3.7.1 (used only by the long-kernel adversary) |
| numpy | 2.5.1 |

Image-wide environment: `CUBLAS_WORKSPACE_CONFIG=:4096:8` (deterministic cuBLAS), `HF_HOME=/data/hf`, `TORCH_HOME=/data/torch`, `TOKENIZERS_PARALLELISM=false`, `PYTHONUNBUFFERED=1`.

### 4.4 How a "slice" is created (the container specification used everywhere)
Every run, native or confined, is one `docker run` of `tfx-workload` with:
- `--gpus device=0`; `--shm-size 2g`; `--network none` (no network in any timed run); `--user <host uid>:<gid>` (non-root); working dir `/work`.
- Mounts: `src/` → `/work` read-only; `data/` → `/data`; a per-run results directory → `/results`; for confined runs `bin/libvgpu.so` → `/opt/hami/libvgpu.so` read-only.
- Environment: `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HOME=/tmp`, `PROBE_OUT=/results/probe.json`; for confined runs `LD_PRELOAD=/opt/hami/libvgpu.so`, `CUDA_DEVICE_MEMORY_LIMIT=<nominal>` (e.g. `4g`, `1728m`), optionally `CUDA_DEVICE_SM_LIMIT=<pct>`; `LIBCUDA_LOG_LEVEL` left at default (1).
- No host-RAM limit except in the one G4 test that needed one (`--memory 1g --memory-swap 1g`).
- Three conditions are used throughout: **native** (no library), **preload** (library loaded, no limit set), **confined** (library + limit).
- Container-level outcome is read from `docker inspect` (`ExitCode`, `OOMKilled`, start/finish times) and the merged stdout/stderr log; HAMi's own `[HAMI-core …]` lines in the log are retained.

### 4.5 Measurement method
- **Ground truth is host-side NVML**, never anything read inside a container. A sampler process on the host polls every **100 ms**: device memory (`nvmlDeviceGetMemoryInfo` v2 struct, so driver-reserved memory is excluded and figures match `nvidia-smi`), GPU/memory utilisation, and per-process used memory (`nvmlDeviceGetComputeRunningProcesses`). Each PID is mapped to its container through `/proc/<pid>/cgroup` (`docker-<id>.scope`).
- **"Host peak"** of a run = maximum over samples of the summed per-process used memory of that container's processes. It **includes the CUDA context (290 MiB)**. All VRAM numbers in this report are host peaks unless labelled otherwise.
- **t90** = seconds from the container's first GPU use to the first sample ≥ 90 % of its peak (how early the peak arrives).
- **Sampler validation** (§5.A): a probe allocates exactly 2048 MiB (`torch.empty(2048·2²⁰, uint8)`) and holds it; the sampler must show a delta of exactly 2048 MiB. It did (290.0 → 2338.0 → 290.0 MiB).
- **In-container view** is also recorded for comparison only: `torch.cuda.mem_get_info()` and in-container NVML, both virtualised by HAMi.
- **Correctness** = SHA-256 over every floating-point tensor of the model's final `state_dict` (fp32 bytes, fixed order) plus the sum of absolute values; for Stable Diffusion the SHA-256 of each output image's uint8 pixel array; for GPT-2 the evaluation perplexity to three decimals. "Bitwise identical" means equal SHA.
- **Determinism** in every workload: seed 0 for Python/NumPy/torch; `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`, TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seeded DataLoader generator and worker seeding.
- **Tolerance rule** (fixed from Phase B before any confined run): if the two native baselines are bitwise identical, a comparison run must match the SHA; otherwise its metric must be within max(3 × baseline spread, 1 %). Throughput loss > 10 % from interception alone triggers stop condition 1; > 3 % is flagged. Any Xid fails the run.
- **Driver health**: `dmesg` (via sudo) is snapshotted before and after every run; any new line containing `Xid` or `NVRM` is recorded. Total across the day: **0**.
- **Throughput** figures come from the workload itself (images/s, examples/s, tokens/s over the training loop, images/s for SD generation excluding model load). **Wall** = container start to exit as seen by Docker.

### 4.6 Workloads (exact configuration)
All run from the same scripts in `src/workloads/`; each writes a JSON with metrics, curve, timeline, parameter SHA.

| id | model / data | training or inference spec | what is checked | native result |
|---|---|---|---|---|
| **resnet18** | torchvision ResNet-18 (CIFAR variant: 3×3 stride-1 conv1, no maxpool), 10 classes, from scratch. CIFAR-10 50 000 train / 10 000 test (from HF `uoft-cs/cifar10`, exported to npz; identical to the Toronto release) | 3 epochs, batch 128 (390 steps/epoch, 1170 total), SGD lr 0.1 momentum 0.9 wd 5e-4 Nesterov, OneCycle (25 % warm-up), fp32; RandomCrop(32, pad 4) + horizontal flip; test batch 256; 4 dataloader workers | test accuracy, final train loss, SHA | acc **0.825**, loss 0.596, 3853 img/s, wall 42 s |
| **resnet50** | torchvision ResNet-50, 10 classes, from scratch. Imagenette2-320 (fast.ai 10-class ImageNet subset): 9469 train / 3925 val JPEGs | 3 epochs, batch 64 (147 steps/epoch, 441 total), SGD lr 0.1 momentum 0.9 wd 1e-4, OneCycle (30 %), CE with label smoothing 0.1, fp32; RandomResizedCrop 224 (scale 0.35–1) + flip; val Resize 255 → CenterCrop 224; 6 workers | val accuracy, loss, SHA | acc **0.499**, loss 1.783, 331 img/s, wall 89 s |
| **bert** | `bert-base-uncased` (110 M) sequence classification. GLUE SST-2 (`nyu-mll/glue`): 16 000 of the 67 349 training sentences (seeded shuffle), full 872-sentence validation set | 1 epoch, batch 32 (500 steps), max length 64 (pad to max), AdamW lr 2e-5 wd 0.01, linear schedule with 10 % warm-up, grad-clip 1.0, fp32 | dev accuracy, loss, SHA | acc **0.906**, 491 ex/s, wall 41 s |
| **gpt2** | `gpt2` (124 M) causal LM. WikiText-2 raw (`Salesforce/wikitext`, `wikitext-2-raw-v1`); train text tokenised, EOS-joined, cut into 512-token blocks (seeded shuffle); test split likewise | 1 epoch, batch 4 blocks (≈1160 steps), AdamW lr 5e-5 wd 0.01, 5 % warm-up, grad-clip 1.0, fp32; eval = exp(mean token NLL) over test blocks in batches of 4, before and after training | eval perplexity, loss, SHA | PPL 49.88 → **21.95**, 19.6k tok/s, wall 136 s |
| **sd** | `stable-diffusion-v1-5/stable-diffusion-v1-5` via diffusers, fp16, DDIM scheduler, safety checker disabled | 4 fixed prompts, 2 per batch, 512×512, 25 steps, per-image CUDA generator seeds 0–3; PNG saved; text encoder → UNet loop → VAE decode | SHA-256 of each image's pixels, images/s, generation seconds | 4 images, **1.20 img/s** (3.3 s generation), load ≈ 12 s, wall 20 s |

Reference numbers quoted in the handout's sense: ResNet-18/CIFAR reaches 93–95 % with a full schedule; Imagenette leaderboards reach 85–93 % at 5+ epochs with tuned recipes; BERT-base SST-2 dev ≈ 92.5 % on full data; GPT-2 small zero-shot WikiText-2 PPL 29.4 (paper, different tokenisation), ≈ 20 fine-tuned. The short schedules here are not meant to reproduce those; they show unambiguous learning (loss falling, accuracy far above 10 % chance, PPL halving) and, more importantly, are **bitwise reproducible**, which is what confinement is judged against.

### 4.7 Synthetic probes (`src/probe/`)
| probe | what it does | used in |
|---|---|---|
| `alloc_hold` | initialise CUDA, wait, allocate exactly N MiB (one uint8 tensor, filled), hold, free, wait; prints timestamps and in-container views | A (sampler validation), D2 (idle cost, N = 0), F3 (multi-process, bypass), G3 (churn, N = 1024) |
| `step_alloc` | allocate 256 MiB chunks until failure, then refine with 64, 16, 2 MiB steps; records exception type, the exact ceiling, whether the process can still compute correctly afterwards (1024-element sum check) and re-allocate after freeing | D1, D4, G2, G4, H (as a job) |
| `burner` | matmul loop for a fixed time: "big" = 4096² fp16 (few large kernels), "small" = 256² fp32 (many tiny kernels); 2 s warm-up, 15 s timed, reports TFLOPS and kernel launches/s | S |
| `adversary` | `oomloop`: allocate 256 MiB chunks to OOM, free, repeat; `longkernel`: one Triton kernel calibrated to ≈ 3 s per launch, back-to-back; `victim`: steady matmul loop holding 1 GiB, meant to be killed from outside | F2 |
| `fork_alloc` | spawn 3 child processes in one container 1.5 s apart, each running `alloc_hold` 1536 MiB | F3 |
| `graph_test` | T1: capture forward+backward of a 1024→4096→1024 MLP with `torch.cuda.CUDAGraph`, replay 20 steps, compare to eager (param diff); T2: capture a graph containing a 6144 MiB allocation + fill + reduction and replay; T4: `torch.compile(mode="reduce-overhead")` on the MLP vs eager | G1 |

### 4.8 Data
All datasets and weights were downloaded once, before any timed run, into `data/` (HF cache 5.2 GB, Imagenette 360 MB, CIFAR npz 177 MB); every measured container ran with `--network none` and the HF offline flags.

### 4.9 Naming and where each number lives
Each run has a name `<phase>_<what>[_r<n>|_m<mult>|_rep<n>]` and a directory `results/<name>/` with `run_meta.json` (everything the harness knows: command, HAMi env, outcome, classification, sampler summary, Xid lines, HAMi log lines, the workload's JSON embedded), `host_sampler.csv` (100 ms samples), `container.log`, and the workload/probe JSON. Phase summaries are `results/phase_<x>_summary.json`. The running log with timestamps is `.agent_logs/2026-09-02_task1_execution.md`. Appendix C lists the file formats.

---

## 5. Results by phase

### 5.A Environment and instrument
**Design.** Capture the stack (§4.1–4.3). Validate the sampler against a known 2048 MiB allocation (native). Establish reproducibility by running every workload twice natively (the Phase B runs double as this).
**Result.**
- Sampler: per-process figure 290.0 MiB (context only) → 2338.0 MiB → 290.0 MiB; delta **exactly 2048.0 MiB**; device-total moved identically; allocate/hold/free phases resolved at 100 ms. Instrument trusted.
- CUDA context (torch 2.13 / cu130 / driver 580): **290 MiB** physical per process.
- NVML v1 `used` on this driver includes ≈ 454 MiB driver-reserved memory (showed 471 MiB "idle"); the v2 struct reports 17 MiB used / 454 reserved, matching `nvidia-smi`. The sampler uses v2.
- Reproducibility (two native runs each): resnet18, resnet50, bert, sd **bitwise identical**; gpt2 metric-identical (PPL 21.951 vs 21.951) but not bitwise (atomic-add ordering in embedding backward). Throughput spread 0.04–0.33 %. This fixed the tolerance rule in §4.5.

### 5.B Native baselines
**Design.** Each workload of §4.6, native condition, twice.
| workload | result | throughput | host peak | t90 | wall |
|---|---|---|---|---|---|
| resnet18 | test acc 0.825 (both) | 3853 / 3841 img/s | **1672 / 1672 MiB** | 0.5 s | 39.2 / 39.3 s (in-container) |
| resnet50 | val acc 0.499 (both) | 330.8 / 331.4 img/s | **6662 / 6662** | 1.2 s | 85.7 / 85.5 s |
| bert | dev acc 0.906 (both) | 490.8 / 490.6 ex/s | **3654 / 3654** | early | 36.9 / 35.7 s |
| gpt2 | PPL 21.951 (both) | 19 580 / 19 563 tok/s | **6730 / 6730** | early | 136.0 / 136.3 s |
| sd | identical image hashes | 1.200 / 1.198 img/s | **4576 / 4576** | at load | 7.7 / 7.6 s generation |

All peaks arrive within the first seconds of GPU use (weights + optimiser state + first batch), so a too-small slice fails fast rather than late. One exception was found the hard way: GPT-2's *evaluation* phase allocates a large logits block late in the run (§6).

### 5.C Build and smoke (interception control)
**Design.** Build as in §4.2. Then every workload in the **preload** condition (library loaded, **no** limit) vs its native baseline.
| workload | correctness vs native | throughput vs native |
|---|---|---|
| resnet18 | bitwise identical | 3821 vs 3853 img/s (+0.8 % slower) |
| resnet50 | bitwise identical | 331.5 vs 330.8 (−0.2 %, noise) |
| bert | bitwise identical | 489.3 vs 490.8 (+0.3 %) |
| gpt2 | PPL identical | 19 555 vs 19 580 (+0.1 %) |
| sd | images bitwise identical | 1.194 vs 1.200 (+0.5 %) |

Interception alone changes nothing and costs nothing measurable (baseline spread is 0.04–0.33 %). Stop condition 1 is clear. Also verified at the probe level: with no limit, in-container `mem_get_info` reports the full 15 921 MiB.

### 5.D Enforcement
**Design.** D1: `step_alloc` under nominal 2g, 4g, 8g and native. D2: context-only process (`alloc_hold` 0 MiB, 14 s) native vs 4g, plus `docker stats` for RSS/CPU. D3: `adversary oomloop` for 30 s in a 4g slice. D4: the three env-var spellings and the per-device form at 3 GiB.
**D1 result.**
| nominal | usable by torch at ceiling | in-container mem_get_info (total / free at context) | in-container NVML at ceiling | **host real peak** | overshoot vs nominal |
|---|---|---|---|---|---|
| 2g | 1792 MiB | 2048 / 1798 | 2040 / 2048 | **2082 MiB** | +34 |
| 4g | 3840 MiB | 4096 / 3846 | 4088 / 4096 | **4128 MiB** | +32 |
| 8g | 7936 MiB | 8192 / 7942 | 8184 / 8192 | **8224 MiB** | +32 |
| native | 15 600 MiB | 15 921 / 15 606 | 16 368 / 16 376 | 15 888 MiB | — |

- **Usable = nominal − 256 MiB.** HAMi charges 250 MiB for the context inside the budget; torch's last small request is rounded to a 20 MiB block, stranding 8 MiB.
- **Real footprint = nominal + ~32 MiB**, constant across sizes, because the context really costs 290 MiB, not 250. Bounded and constant, so not a bypass; a pool must debit nominal + 64 MiB physical per slice.
- In-slice NVML is virtualised (2040/2048 shown vs 2082 real) — confirms "measure from the host".
- Overrun → `torch.cuda.OutOfMemoryError` ("GPU 0 has a total capacity of 4.00 GiB of which 8.00 MiB is free …"); the process survives, computes correctly afterwards (exact-sum check), frees and re-allocates. **Catchable OOM, no driver fault.**
**D2 result.** Context-only process: host GPU memory **290 MiB native, 290 MiB HAMi** (HAMi adds 0); container RSS 323.7 vs 327.4 MiB (+4 MiB); idle CPU 0.12 % vs 1.32 % of a core (HAMi's utilisation-watcher thread).
**D3 result.** 459 allocate-to-OOM cycles in 30 s under 4g: 459 `OutOfMemoryError`, 0 other errors, 0 Xid.
**D4 result.** `CUDA_DEVICE_MEMORY_LIMIT_0=3g`, `3072m`, and `3221225472` all give total 3072 / usable 2816 MiB.

### 5.E Real workloads under a ceiling
**Design.** E1: each workload confined at the smallest "catalogue" tier (2048/4096/8192/12288 MiB) ≥ 1.15 × its native host peak. E2: headroom search — limit = roundup64(peak × m) for m ascending through 1.0, 1.03, 1.06, 1.1, 1.15, 1.2, 1.3, 1.5; the first m that completes is confirmed with two more runs (3/3 required). E3 (added): limits *below* the peak at −64, −192, −512 MiB for resnet18, bert, sd. Every run compared to the native baseline with the §4.5 rule; throughput compared to native and to preload to separate the cost of being intercepted from the cost of being constrained.
**Result.**
| workload | native peak | E1 tier | tightest passing slice (m = 1.0) | correctness (all runs) | throughput native → preload → confined |
|---|---|---|---|---|---|
| resnet18 | 1672 | 2048m | 1728m, 3/3 | bitwise identical | 3853 → 3821 → 3825 img/s |
| resnet50 | 6662 | 8192m | 6720m, 3/3 | bitwise identical | 330.8 → 331.5 → 330.5 img/s |
| bert | 3654 | 8192m | 3712m, 3/3 | bitwise identical | 490.8 → 489.3 → 488.8 ex/s |
| gpt2 | 6730 | 8192m | 6784m, 3/3 | PPL 21.951 identical | 19 580 → 19 555 → 19 521 tok/s |
| sd | 4576 | 8192m | 4608m, 3/3 | images bitwise identical | 1.200 → 1.194 → 1.195 img/s |

- 20 of 20 confined runs matched the baseline. Cost of being *constrained* on top of being intercepted: ≤ 0.1 %, not measurable.
- Host peak under confinement equals the native peak in every case: PyTorch's caching allocator did not fragment or grow differently.
- The first multiplier tried (1.0) passed everywhere, so 1.0× is an upper bound on the true requirement; E3 shows the margin behind it:

| E3 (below peak) | peak − 64 | peak − 192 | peak − 512 |
|---|---|---|---|
| resnet18 (1672) | ok, host 1672 | ok, host 1552 | **ok, host 1258 (−25 %), wall unchanged** |
| bert (3654) | ok, host 3654 | ok, host 3462 | **clean CUDA OOM after 6.3 s** (first batch) |
| sd (4576) | ok, host 4576 | ok, host 3896 | ok, host 4024 |

The native peak includes allocator cache that PyTorch returns under pressure, so true floors are 5–25 % below the measured peak with no throughput penalty; a slice that is genuinely too small fails within seconds, cleanly.

### 5.F Co-tenancy
**Design.** F1: two containers launched 2 s apart under one host sampler, each in its E1 tier; compared to the solo E1 run for slowdown and to the sum of solo walls for makespan. Tiers made three heavy pairs not fit (8192 + 8192 > capacity), so **F1b** re-ran them with *tight* slices = tightest passing slice + 256 MiB. F2: victim = resnet18 in 2048m (F1b: its tier), adversary started 5 s later in a 4g slice: `oomloop` 75 s; `longkernel` 75 s; `victim` mode hard-killed with `docker kill -9` at 25 s; and `oomloop` **unconfined** (no library) 75 s. F3: `fork_alloc` 3 × 1536 MiB in one 4g slice; then bypass characterisation: `env -u LD_PRELOAD` inside a 4g slice trying to allocate 6144 MiB, with and without `/etc/ld.so.preload` (read-only bind, containing `/opt/hami/libvgpu.so`).
**F1 / F1b result** (every job bitwise/metric-correct, 0 Xid):
| pair (slices) | slowdown A / B vs solo | makespan concurrent vs back-to-back | resident VRAM |
|---|---|---|---|
| resnet18 + resnet18 (2048m + 2048m) | 1.98× / 1.95× | 85.4 s vs 85.0 s (0.996×) | 3.4 GB |
| resnet18 + bert (2048m + 8192m) | 1.88× / 1.97× | 81.2 s vs 82.6 s (1.02×) | 5.3 GB |
| bert + gpt2 (3968m + 7040m) | 1.81× / 1.19× | 170.5 s vs 181.3 s (1.06×) | 10.4 GB |
| resnet50 + sd (6976m + 4864m) | 1.06× / **3.41×** | 95.0 s vs 109.1 s (1.15×) | 11.3 GB |
| gpt2 + sd (7040m + 4864m) | 1.03× / **2.92×** | 145.5 s vs 161.3 s (1.11×) | 11.3 GB |
| resnet50 + gpt2 (6976m + 7040m) | 1.96× / 1.57× | 224.4 s vs 230.3 s (1.03×) | 13.4 GB |

Running two jobs together finishes both only 0–15 % sooner than back-to-back: saturating trainers divide the card almost exactly (per-job fractions sum ≈ 1.0); heterogeneous pairs recover a little idle time. Sharing buys *capacity* (N students on one card at ~1/N speed), not speed. Inference with many short kernels (Stable Diffusion) loses most under time-slicing next to a saturating trainer (3×), which matters for interactive use.
**F2 result** (victim resnet18; all victim runs bitwise identical to its solo run):
| adversary | adversary outcome | victim correctness | victim slowdown |
|---|---|---|---|
| allocate-to-OOM loop (4g slice) | 1096 OOM cycles in 75 s, clean exit | bitwise identical | 1.04× |
| long single kernels (Triton, ≈ 3.1 s each, 22 kernels) | clean exit | bitwise identical | 1.95× |
| hard `docker kill -9` at 25 s | exit 137 | bitwise identical | 1.31× while alive |
| **unconfined** OOM loop (whole card to 15.6 GB, 283 cycles) | clean exit | bitwise identical | 1.04× |

No Xid, no reset, no wrong result. Long kernels are the only neighbour behaviour that meaningfully hurts (contention, not a breach). The unconfined case passed only because the victim had already allocated its peak; a later-allocating victim would be starved by an unconfined neighbour, so every tenant must be confined.
**F3 result.** Children 1 and 2 (1536 MiB each) succeed, child 3 gets `OutOfMemoryError`: 3072 + 3 × 250 = 3822 ≤ 4096. **Per slice, not per process**; fork does not bypass.
**Bypass characterisation.** `env -u LD_PRELOAD python …` in a "4g" slice: allocated 6144 MiB (host saw 6434 MiB), in-container total 15 921 MiB — **the env-var route is decorative** against a tenant who edits their environment. With `/etc/ld.so.preload` bound read-only and `LD_PRELOAD` unset: 6 GiB refused, total 4096 MiB. Residual: a root tenant with a statically linked CUDA binary or private loader is outside any preload mechanism; this is inherent to the userspace approach.

### 5.G Failure modes
**Design.** G1: `graph_test` (§4.7) in four conditions: native / default allocator, HAMi 4g / default allocator, HAMi 4g / `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`, native / async. G2: resnet18 in 4g killed with `docker kill -9` at t = 40 s; NVML polled at 100 ms for memory release; then `step_alloc` in a fresh 8g slice immediately. G3: 15 back-to-back cycles of container create → `alloc_hold` 1024 MiB (hold 1 s) → destroy in a 4g slice. G4: three OOM kinds: a NumPy host-RAM hog under `--memory 1g` (no swap), `step_alloc` under 2g, and resnet50 training in a 2g slice.
**G1 result.**
| allocator | HAMi 4g | T1 graph train step vs eager | T2 6 GiB alloc inside capture | T4 compile reduce-overhead |
|---|---|---|---|---|
| default | no | ok, param diff 0 | ok (6468 MiB reserved) | ok |
| default | **yes** | ok, param diff 0 | **refused: OutOfMemoryError, "total capacity 4.00 GiB"** | ok |
| cudaMallocAsync | no | ok | ok (graph mem nodes, 192 MiB reserved after) | fails (torch: checkPoolLiveAllocations unsupported) |
| cudaMallocAsync | **yes** | **fails: `AcceleratorError: CUDA error: invalid argument`** | refused ("would exceed allowed memory, device limit 4 GiB") | fails (same torch limitation) |

The failing path is HAMi-core's own: its `cuMemAllocAsync` accounting derives a garbage device id during stream capture (`Illegal device id: 32517`) and `cuDeviceGetMemPool` returns `CUDA_ERROR_INVALID_VALUE`. A HAMi-core bug affecting only the opt-in async allocator. With the default allocator, CUDA Graphs and `torch.compile` reduce-overhead work and stay clamped; no graph-based clamp bypass exists (the suspected unhooked graph-node path is refused before capture completes). The plan's issue #1360 turned out to be a torch.compile-under-debugger error, closed upstream as unrelated to HAMi; the CUDA-graph question was tested directly instead. In eager mode (§5.S4) the async backend still trains correctly and respects the ceiling, but logs the same illegal-device-id error and a `cuMemoryAllocate failed res=201`, with PyTorch's allocator recovering by retry — it works by accident.
**G2 result.** Exit 137 within 0.22 s; **GPU memory back to idle (1697 → 17 MiB) in 0.22 s**; 0 Xid; the fresh 8g slice got its full 7936 MiB usable.
**G3 result.** 15/15 ok, **3.77 s per cycle** (container start + CUDA init dominate), host idle 16.9 MiB before and after, no leaked processes, no HAMi errors, 0 Xid.
**G4 result.**
| case | signature seen from the host |
|---|---|
| container RAM OOM (`--memory 1g`) | `State.OOMKilled=true`, exit 137, empty stdout |
| CUDA OOM in a probe | caught in-process, exit 0, process continues |
| CUDA OOM in training (resnet50 in 2g) | `torch.cuda.OutOfMemoryError` at the first batch, 4 s in; exit 3 + JSON status `cuda_oom` from the workload wrapper (raw torch: exit 1, traceback in stderr) |

`OOMKilled` → host RAM; exit 137 without it → killed/preempted; "OutOfMemoryError" in stderr → the tenant outgrew its slice. All three are distinguishable from the host alone.

### 5.S Compute limiting (`CUDA_DEVICE_SM_LIMIT`)
**Design.** S1: `burner` (both regimes, 15 s) at native, and in a 4g slice with SM limit 100, 75, 50, 25, 10. S2: resnet18 in 2048m at SM 50 and 25. S3: resnet18 uncapped + resnet18 at SM 30, both 2048m, concurrent. S4: async-allocator eager check (`step_alloc` in 4g, resnet18 in 2048m, `backend:cudaMallocAsync`).
**Result.**
| SM limit | large fp16 matmuls (TFLOPS, ×native) | small fp32 kernels (launches/s, ×native) | ResNet-18 training (img/s, ×native) |
|---|---|---|---|
| native / 100 | 100.2 / 100.5 (1.00) | 67.6k / 64.4k (1.00 / 0.95) | 3853 (1.00) |
| 75 | 97.2 (0.97) | 60.1k (0.89) | — |
| 50 | 92.0 (**0.92**) | 64.8k (**0.96**) | 2082 (**0.54**), wall 1.79× |
| 25 | 74.1 (0.74) | 36.2k (0.54) | 968 (0.25), wall 3.75× |
| 10 | 31.7 (0.32) | 14.3k (0.21) | — |

- The percentage is **not a share of compute**: a matmul stream at "50 %" keeps 92 % of the card; the limiter bites hard only below ≈ 25 %. ResNet-18, a mix of many small kernels and sync points, is throttled almost proportionally. The same number means different things to different workloads, as the handout warned.
- Results stay bitwise correct under the cap (resnet18 at 50 % and 25 %: same SHA, acc 0.825).
- As a **priority mechanism it works**: uncapped resnet18 next to resnet18 at 30 % ran at 1.47× its solo time instead of the 1.98× of an equal pair; the capped one at 3.07×. An owning lab's job can be favoured over a guest's.
- It is **not work-conserving**: after the uncapped job finished (62.6 s) the capped job kept sleeping on an idle card; pair makespan 133 s vs 85 s for the equal pair. Use only for priority under contention, or accept wasted capacity.
- S4: async allocator in eager mode hits the ceiling correctly (3840 MiB usable) and trains resnet18 to the correct SHA at 3806 img/s, but HAMi logs `Illegal device id` / `cuMemoryAllocate failed res=201` on the way (see §5.G).

### 5.H The runner
**Design.** `src/runner/tfx_runner/`: `GpuPool` (capacity = total − 256 MiB reserve; a slice of nominal N debits N + 64 MiB), FIFO queue with backfill (`pick_next` is the scheduler plug-point), Docker launcher mounting `libvgpu.so` and setting per-slice limits (§4.4), host-side NVML `Watcher` per job (PIDs mapped through cgroup; peak, time-of-peak, over-limit flag, timeout kill), append-only JSONL `Ledger`, `classify_exit`. CLI `python -m tfx_runner run jobs.yaml --concurrency 2`. Soak: six-job batches (resnet18 1984m, bert 3968m, gpt2 7040m, sd 4864m, resnet50 6976m, `step_alloc` 2048m — the tight slices), concurrency 2, submitted repeatedly for 3 hours; after each batch the pool and host NVML were checked.
**Result.**
| | |
|---|---|
| duration / batches / jobs | 3.01 h / 34 / **204 submitted, 204 finished, 204 ok** |
| outputs verified against baselines | 34/34 for each job type: parameter SHA (resnet18, bert, resnet50), image SHA (sd), PPL 21.951 (gpt2), ceiling 1792 MiB + correct post-OOM compute (probe) |
| host-measured peak per job | identical across all 34 repeats and equal to the Phase B baseline; 0 over-limit observations (margin 64 MiB) |
| capacity after each of 34 batches | 16 120 / 16 120 MiB (back to full every time) |
| host GPU memory before / after 3 h | 16.9 / 16.9 MiB; 0 leaked processes; 0 Xid; 408 pool alloc/release events; 612 ledger lines |
| wall (median, two jobs resident) | resnet18 80 s, bert 81 s, sd 58 s, gpt2 234 s, resnet50 176 s, probe 2.7 s |

No leaked memory, no stranded capacity, no wrong result over three hours of continuous churn. A first attempt with the coarse 8192m tiers showed the pool correctly refusing to co-schedule two nominal 8 GB slices (§2); the soak was restarted with tight slices so that two jobs were resident throughout.

---

## 6. Deviations from the plan (with reasons)
- **Host driver 580.178.04 / CUDA 13.0** is newer than the plan assumed; matched with `pytorch/pytorch:2.13.0-cuda13.0` and HAMi-core's CUDA 13.3 build container. Worked first time.
- **HAMi-core has no tags**; pinned the 2026-08-31 main commit `f01e9f23` as a submodule and built with upstream's own recipe (Debug build type is what upstream ships).
- **No `nvidia` Docker runtime** is registered; `--gpus` works through the toolkit hook, nothing was reconfigured.
- **CIFAR-10** from the Hugging Face mirror (`uoft-cs/cifar10` → npz); the Toronto server served at ≈ 60 kB/s. Same data.
- **GPT-2 sizing**: batch 8 × 512 fp32 overflowed the 16 GB card *natively* during evaluation (a 1.5 GiB logits block against a fragmented cache; training had run 500+ steps). Reduced to batch 4, eval batch 4, cache emptied before eval. A workload-sizing mistake, not a HAMi effect; also a reminder that late large allocations (evaluation, checkpointing) exist even when training peaks early.
- **Headroom search** started at 1.0× and every workload passed there; E3 (below-peak probe) was added so the figure is not just an upper bound.
- **Catalogue tiers in F1** pushed every mid-size workload into 8192m, so three heavy pairs did not fit and were skipped; re-run as F1b with tight slices.
- **`/etc/ld.so.preload` injection** was tested in addition to the plan's `LD_PRELOAD` after the env route proved bypassable.
- **Issue #1360** is not a CUDA Graphs failure (closed upstream as a torch.compile/debugger error); CUDA Graphs were tested directly. **vLLM itself was not run** (large dependency outside the workload list); the vLLM issue's behaviour is inferred from the PyTorch-level tests.
- **SM-limit sweep** was initially omitted and run afterwards when the omission was pointed out.
- **Reference numbers** are used as sanity checks (learning is unambiguous), not reproduced with full schedules; correctness rests on bitwise comparison.
- **Workspace layout**: `data/` and `results/` were added alongside the prescribed folders.

## 7. What breaks / what the platform must handle
1. **Inject via `/etc/ld.so.preload` (read-only bind), never via `LD_PRELOAD` alone.** Run tenants non-root; prefer a read-only rootfs. Static binaries / private loaders are outside any preload mechanism.
2. **Size slices as `roundup64(peak) + 64 MiB` and debit `slice + 64 MiB` physical.** Usable inside is `slice − 256 MiB`; publish the usable figure. Keep a ≈ 256 MiB pool reserve.
3. **Per-slice fixed cost is one CUDA context (290 MiB physical) plus 256 MiB of nominal budget.** Four nominal 4 GB or two nominal 8 GB slices do not fit a 16 GB card (§2 table).
4. **Co-tenancy shares, it does not accelerate.** Two saturating jobs each run at ≈ ½ speed; sell capacity and predictability. Long-running kernels from a neighbour are the worst contention (≈ 2×); OOM-looping neighbours barely register (1.04×); short-kernel inference next to a trainer loses ≈ 3×.
5. **CUDA OOM inside a slice is a clean Python exception**; the process survives; the driver is unaffected after > 1000 OOMs. Tell users the "total capacity" in the message is their slice.
6. **Do not offer `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`** under HAMi (graph capture fails; eager only works by retry). Default allocator + CUDA Graphs + `torch.compile` is fine. Worth reporting upstream: `allocator.c:315 cuDeviceGetMemPool` with an illegal device id during capture.
7. **Hard kills are safe and fast** (memory back in ≈ 0.2 s). Container churn costs ≈ 3.8 s per job; budget for it.
8. **Three OOM signatures must be reported differently**: `OOMKilled` (host RAM), exit 137 without it (killed), `OutOfMemoryError` in stderr (slice too small).
9. **In-slice `nvidia-smi`/NVML report the slice, by design.** Monitoring, billing and debugging must use host-side NVML with PID→container mapping.
10. **`CUDA_DEVICE_SM_LIMIT` is not a share of compute** (50 % → 92 % of the card for big kernels, 54 % for ResNet-18) and is not work-conserving. Usable as a priority lever under contention; do not present it as a percentage of performance.

---

## Appendix A. Reproduction
```
git submodule update --init                       # HAMi-core at f01e9f23
cd src/hami/HAMi-core && make build-in-docker && cp build/libvgpu.so ../../../bin/
docker build -t tfx-workload -f src/docker/Dockerfile.workload src/docker
docker run --rm --gpus device=0 -v $PWD/src:/work:ro -v $PWD/data:/data -w /work/workloads tfx-workload python prepare_data.py   # once, online
venv/bin/python src/experiments/phase_d.py                     # probes only
venv/bin/python src/experiments/phase_bc.py b && venv/bin/python src/experiments/gates.py b
bin/run_chain.sh                                               # C → gate → E → gate → F → gate → G
venv/bin/python src/experiments/phase_e3.py; venv/bin/python src/experiments/phase_f1b.py
venv/bin/python src/experiments/phase_s.py
venv/bin/python src/experiments/phase_h.py --hours 3 --concurrency 2
venv/bin/python src/experiments/run_job.py --name x --hami --mem-limit 4g --cmd "python /work/probe/step_alloc.py"   # any single run
```
`gates.py <phase>` exits 2 on a handout §6 stop condition, which halts `run_chain.sh`. Total machine time for everything above: ≈ 6 h (3 h of it the soak).

## Appendix B. Timeline (2026-09-02, UTC)
13:07 go · 13:13 HAMi-core built · 13:37 sampler validated · 13:52–13:58 Phase D · 14:03–14:20 Phase B · 13:43–14:26 chain (gpt2 re-run, C, E, F, G, each gate CONTINUE) · 14:27–14:42 E3 + F1b · 14:43–17:44 Phase H soak · 18:0x–18:2x Phase S.

## Appendix C. Raw data and formats
- `results/<run>/run_meta.json`: `name, tag, cmd, hami (env dict or null), env, cgroup_mem, libvgpu_sha256, t_start_epoch, host_before/host_after (NVML snapshot), wall_s, sampler (summary below), xid_events, classification, hami_log_lines, outcome {exit_code, oom_killed, error, started_at, finished_at}, container_id, container_results {<file>.json: …}`.
- Sampler summary: `baseline_used_mib, driver_reserved_mib, peak_total_used_mib, peak_total_t, peak_total_minus_baseline_mib, containers {<cid>: {peak_mib, peak_t, first_seen_t, last_seen_t, lifetime_s, t_to_90pct_peak_s}}`.
- `results/<run>/host_sampler.csv`: `t, used_mib, util_gpu, util_mem, procs(pid:cid:mib;…)` at 100 ms.
- Workload JSON: `name, status (ok|cuda_oom|error), error, args, env {torch, cuda, cudnn, device, env{LD_PRELOAD, CUDA_DEVICE_MEMORY_LIMIT, CUDA_DEVICE_SM_LIMIT, PYTORCH_CUDA_ALLOC_CONF}, mem_get_info_*}, wall_s, metrics, curve, torch_max_allocated_mib, torch_max_reserved_mib, timeline [(t, step, allocated, reserved)], param_sha256, param_abs_sum`.
- Probe JSON (`probe.json`): probe-specific; `step_alloc` has `steps, first_failure, refined_failure_<n>, ceiling_allocated_mib, post_oom_compute, realloc_after_free`; `burner` has `big/small {iters, seconds, kernels_per_s, tflops}`.
- Phase summaries: `results/phase_d_summary.json, phase_e_summary.json, phase_e3_summary.json, phase_f_summary.json, phase_f1b_summary.json, phase_g_summary.json, phase_s_summary.json, phase_h_summary.json`; runner ledger `results/runner_phase_h/ledger.jsonl` and per-job dirs.
- Upstream material saved for reference: `references/hami/` (HAMi-core README at the pinned commit, HAMi issue #1360 with comments, vLLM issue #40937).

## Appendix D. Glossary
- **native** — no HAMi library loaded. **preload** — library loaded via `LD_PRELOAD`, no limit set. **confined** — library + `CUDA_DEVICE_MEMORY_LIMIT`.
- **nominal** — the configured limit. **usable** — what a framework can allocate inside (nominal − 256 MiB). **physical / real footprint** — what host NVML sees for the container's processes (nominal + ~32 MiB at the ceiling; includes the 290 MiB context).
- **host peak** — max summed per-process VRAM of one container, host NVML, 100 ms sampling. **t90** — time from first GPU use to 90 % of that peak.
- **tier / catalogue slice** — 2048/4096/8192/12288 MiB, smallest ≥ 1.15 × peak. **tight slice** — tightest passing E2 slice + 256 MiB.
- **slowdown** — wall time in the pair ÷ wall time of the same job alone in the same slice. **makespan** — start of the first job to end of the last. **back-to-back** — sum of the two solo walls.
- **bitwise identical** — equal SHA-256 over all final floating-point parameters (or image pixels).
- **Xid** — NVIDIA driver error event in the kernel log; any occurrence would have failed the run. None occurred.
