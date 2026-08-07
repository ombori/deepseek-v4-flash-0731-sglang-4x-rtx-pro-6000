# DeepSeek-V4-Flash-0731 on 4× RTX PRO 6000 Blackwell (SM120) — SGLang recipe

This is the config we run in production: [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (284B MoE / 13B active, native FP4+FP8, 1M context) on four RTX PRO 6000 Blackwell cards. TP4, DP4 attention, EP4, DSPARK speculative decoding at draft depth 4, fp8 KV cache.

Stock sglang v0.5.16 can't serve this. DSPARK crash-loops at warmup on SM120, several verify and tool-call-streaming bugs bite under real traffic, and the checkpoint's default speculative draft depth silently corrupts output (that last one also affects upstream `main`, details below). This repo has the patch set, the image build, the serving config, the measured numbers, and the failure boundaries we mapped so you don't have to.

Everything was measured on exactly this rig, but parts of it travel further than the name suggests: the patches apply to any SM120 card count, the draft-depth corruption affects any DSpark deployment on any engine, and the dependency pins hold for Blackwell workstation cards generally. See [What here generalizes](#what-here-generalizes).

The existing public recipes for this model target 2× GPU / TP=2 (see [Related work](#related-work)). This one is the 4× / TP=4 / DP-attention shape.

## What this recipe optimizes for

High-concurrency agentic serving: many parallel sessions, tool calling, long contexts. That objective drove the biggest layout choice here — DP attention gives ~4x the KV capacity (MLA's KV cannot be sharded under plain TP, so every added replica of it multiplies concurrent context) and the aggregate numbers below, at a deliberate cost to single-stream latency. On this config we measure ~2,300-2,400 out tok/s at 64 concurrent and 78-98 tok/s at batch size 1.

If your workload is a single interactive stream, there is real headroom on this hardware that this recipe leaves on the table. Our own non-DP TP4 runs measured ~90-98 tok/s at bs=1, TP=2 does better still per stream (fewer PCIe hops; bs=1 decode on these boxes is interconnect-latency bound, not bandwidth bound), and layer-split runtimes avoid the per-layer TP synchronization entirely — llama.cpp with the official GGUF and an MTP draft reaches 100-130 tok/s on two of these cards at 512K context. Each of those trades away something this recipe keeps: concurrency, the full 1M context, or the native-precision checkpoint. Pick the shape that matches your traffic; the numbers in this README only claim to be optimal for ours.

## 2026-08-06 update: faster prefill

The patch set now carries two additions past the published image: a backport of open PR [#29927](https://github.com/sgl-project/sglang/pull/29927) (the SM120 prefill stack) and a completion patch that puts the #32183 backport on the code path v0.5.16 actually executes. Together with an sgl-deep-gemm bump in the Dockerfile (0.1.4.post1 → 0.1.5.post1, the version #29927 was developed against) and three env vars in the compose file, single-request prefill improves 19–25% at 256–512K contexts and TTFT drops 16–20%. Decode is unchanged (bs=1 within 4%, conc64 +2%).

| context, single request | published image | this repo |
|---|---|---|
| 64K | 10.7 s TTFT / 5,747 tok/s | 9.7 s / 6,287 |
| 256K | 61.1 s / 4,234 | 48.8 s / 5,283 |
| 512K | 143.7 s / 3,626 | 120.5 s / 4,315 |

The updated stack passed the same ship gate as the published image: 5 cold launches under the bursty repro protocol below, 0 corruption events across 11,187 requests, 8/8 streaming truncation probes complete.

The prebuilt image now includes all of this: tags `:latest` and `:2026-08-06.2`, digest `sha256:b4f21873370080c5b996d38cb82d6005aae9c10e50cda644b2ecbf912b0029aa`. The previous image remains at `:2026-08-06` — note the new compose/env recipe aborts at boot on that older build (its DeepGEMM predates SM120 support), so keep compose and image in step.

One thing the DeepGEMM bump does not unlock: its FP4 grouped MoE GEMM still fails on SM120 inside DeepGEMM's own warmup (CUDA 719 at DSv4 per-rank shapes), so the MoE runner stays `flashinfer_mxfp4` and the remaining piece of #29927's recipe waits on a DeepGEMM fix.

## Run the prebuilt image

The image is public, so there's no build step:

```sh
docker pull ghcr.io/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000:latest
```

`:latest` tracks `main`; date tags (e.g. `:2026-08-06`) pin a specific build. It's the exact build the numbers under [Measured performance](#measured-performance) came from.

The weights are not in the image. Download the ~158 GB checkpoint and mount it read-only; the commands below assume `/models/DeepSeek-V4-Flash-0731` on the host:

```sh
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
    --local-dir /models/DeepSeek-V4-Flash-0731
```

Minimal `docker run` with the validated shape. The flags mirror [`docker-compose.example.yml`](./docker-compose.example.yml); that file is canonical if the two ever differ, and it documents why each flag is set:

```sh
docker run -d --name sglang \
  --gpus '"device=0,1,2,3"' --ipc=host --shm-size 32g \
  -p 8000:30000 \
  -v /models:/models:ro \
  -v "$PWD/sglang-cache:/root/.cache" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS=1024 \
  -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 \
  -e SGLANG_OPT_USE_TILELANG_INDEXER=0 \
  -e SGLANG_OPT_DEEPGEMM_HC_PRENORM=1 \
  --entrypoint python3 \
  ghcr.io/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000:latest \
  -m sglang.launch_server \
  --model-path /models/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 --port 30000 \
  --tp-size 4 --dp-size 4 --enable-dp-attention --enable-dp-lm-head \
  --ep-size 4 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --context-length 1048576 \
  --swa-full-tokens-ratio 0.1 \
  --max-running-requests 256 --cuda-graph-max-bs 64 \
  --kv-cache-dtype fp8_e4m3 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algorithm DSPARK \
  --speculative-attention-mode decode \
  --speculative-dspark-block-size 4 \
  --disable-custom-all-reduce \
  --chunked-prefill-size 4096 \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --enable-cache-report
```

For anything long-lived, use the compose file instead (healthcheck, restart policy, JIT-cache persistence notes). It references the local build tag, so retag the pulled image first:

```sh
docker tag ghcr.io/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000:latest \
    sglang:v0.5.16-dsv4-sm120
docker compose -f docker-compose.example.yml up -d
```

First boot takes 10–15 minutes (JIT kernel compile plus flashinfer autotune); the compose healthcheck allows for it. The OpenAI-compatible API lands on `:8000`.

Send `temperature 1.0 / top_p 1.0` from clients. That's DeepSeek's calibration for this checkpoint; lower temperatures drive repetition loops.

## Hardware

| | |
|---|---|
| GPUs | 4× NVIDIA RTX PRO 6000 Blackwell Max-Q, 96 GB (SM120 / GB202) |
| Interconnect | no NVLink; each GPU on its own PCIe gen5 x16 root port, all TP traffic crosses the CPU fabric |
| Host RAM | ≥ 128 GB recommended (the ~158 GB checkpoint is mmap'd) |
| Driver / CUDA | 595-server-open / CUDA 13.0 (`cu130` base image) |

On this topology TP=4 wins aggregate throughput and TP=2 wins bs=1 latency (fewer PCIe hops). bs=1 decode is interconnect-latency bound, not bandwidth bound. EP4 helps bs=1 by shrinking the MoE all-reduce. `nvidia-smi topo -m` shows `NODE` for every pair on this box.

## Building the image yourself

Equivalent to pulling the prebuilt one; the build is deterministic given the pinned base image and patch set. `.github/workflows/build-image.yml` automates build+push but needs a self-hosted runner with ~100 GB free disk. The ~46 GB image does not fit GitHub-hosted runners.

```sh
git clone https://github.com/ombori/deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000
cd deepseek-v4-flash-0731-sglang-4x-rtx-pro-6000
docker build -t sglang:v0.5.16-dsv4-sm120 .

# put the DeepSeek-V4-Flash-0731 checkpoint under /models, then:
docker compose -f docker-compose.example.yml up -d
```

## Measured performance

`sglang.bench_serving`, random 1024-in / 512-out, measured 2026-08-05 on the exact image and compose in this repo (only the `--speculative-*` flags removed for the no-spec column). Numbers are output tok/s. The [concurrency sweep](#concurrency-and-operating-points) below is a later run of the same config, so its absolute figures differ by a few percent (run-to-run variance on a thermally soaked box); use the sweep for scaling shape and this table for spec-vs-no-spec.

| workload | no spec | DSPARK γ=4 | Δ |
|---|---|---|---|
| conc 16 | 619 | **1196** | +93% |
| conc 16 mean TTFT | 955 ms | **408 ms** | −57% |
| conc 16 mean ITL | — | **11.9 ms** | |
| conc 64 | 1511 | **2297** (mean ITL 25 ms) | +52% |
| bs=1 decode | ~62 | **78–90** | |

One caveat you should not skip: synthetic prompts flatter spec decode. Random-token prompts drive the model into highly predictable continuations, which DSPARK predicts almost perfectly. We measured accept length 4.4–4.7 of 5 draft tokens on random prompts versus ~2.1 on real prose, so treat the decode-bound figures above as roughly 2× optimistic for real agentic traffic. The bs=1 essay-prompt measurement (~85 tok/s) is the more honest single-stream number.

Accept length at γ=4 is 2.0–2.2 of 5 verify rows on real text, and it stays ~2.2 at every depth we tested (3, 4, 6, 7). The draft head realises about the same number of tokens regardless of window, so wider windows only dilute the accept rate and cost bs=1 latency. γ=4 measured best at bs=1 with aggregate within ~4% of the widest window.

Prefill runs ~5.4K tok/s at 90K ctx and stays roughly flat to large context with the indexer-threshold env set (see the compose file). Without it, prefill decays to ~2K tok/s at large context because every DP prefill chunk falls back to a paged-gather kernel.

## Concurrency and operating points

Concurrency sweep on this exact image, 1024-in / 512-out unless noted. Random dataset, so the synthetic-prompt caveat applies to the absolute decode numbers; the scaling shape is the point here.

| C | out tok/s | total tok/s | TTFT mean | TTFT P99 | ITL mean |
|---|---|---|---|---|---|
| 1 | 171 | 513 | 200 ms | 228 ms | 5.5 ms |
| 8 | 767 | 2,300 | 343 ms | 1.0 s | 9.4 ms |
| 16 | 1,171 | 3,513 | 443 ms | 1.3 s | 12.2 ms |
| 32 | 1,734 | 5,201 | 448 ms | 1.2 s | 16.9 ms |
| 64 | 2,467 | 7,401 | 565 ms | 2.2 s | 23.7 ms |
| 96 | 2,892 | 8,677 | 684 ms | 3.2 s | 30.4 ms |
| 128 | 3,293 | 9,880 | 777 ms | 3.7 s | 35.6 ms |
| 192 | 3,370 | 10,110 | 2.03 s | 12.9 s | 51.0 ms |
| 256 | 3,316 | 9,948 | 4.52 s | 17.1 s | 65.7 ms |

The knee is at C≈128: 96→128 still scales (+13.9% throughput), past 128 you buy ≤2.3% more throughput for 2.6–5.8× the TTFT. It's compute-bound, not cap-bound. The TTFT inflection sets in below `--max-running-requests 256`, and at C=256 the cap only just engages (measured running concurrency 246.6), so raising the cap or `--cuda-graph-max-bs` buys zero throughput and only deepens internal queues. We run C=32–64 for interactive agent work (ITL 17–24 ms, TTFT ≤ 0.6 s) and C=128 for max aggregate.

Long context moves the knee. At 8192-in / 512-out the regime flips to prefill-bound (aggregate prefill tops out around 9–11K tok/s) and the knee collapses to C ≤ 32: C=32 gives TTFT 4.0 s mean / 18 s P99, C=128 gives 11.1 s mean / 70 s P99. Median ITL holds at 10–18 ms but P95 spikes to 0.3–0.9 s as decode stalls behind prefill chunks (`--chunked-prefill-size 4096` is 1024 per DP rank). Practically: keep effective concurrency ≤ 32 for cold large contexts. A cold 50–270K-token context costs tens of seconds of TTFT at the prefill ceiling no matter what; warm sessions fare better via radix-cache prefix hits.

## Known issues

| # | issue | status / workaround |
|---|---|---|
| 1 | **DSPARK draft depth γ=5 corrupts output on SM120 — and only γ=5.** At γ=5 generations garble under real load: token loops (`` ` for ` for ` for `` …), word salad, `</think>` leaking into content, and a broken spec-decode greedy invariant (temp-0 with spec ≠ temp-0 without, same image/shape). γ=3, 4, 6 and 7 are all clean under the same protocol — the fault is **not** depth-monotonic. γ=5 is exactly the checkpoint's native `dspark_block_size`, and it corrupts whether it is set explicitly or inferred. Reproduced on patched v0.5.16 **and** on upstream `main` @ `211ee642` (2026-08-05). Filed upstream as [sglang#33800](https://github.com/sgl-project/sglang/issues/33800). Full sweep + repro protocol below. | run an explicit `--speculative-dspark-block-size 4` (this repo's compose). **Never omit the flag** — omitting it infers γ=5 from the checkpoint, i.e. the one broken value |
| 1b | **Assistant prose is truncated mid-sentence when a tool call follows (streaming only).** The DeepSeek streaming detector drops normal text that shares a delta with the DSML tool-call opener; speculative decoding makes multi-token deltas common, so it fires often. Measured on the unpatched image with a preamble-forcing prompt: **4/8 streaming responses cut mid-sentence** (sometimes empty), while non-streaming was 8/8 complete. | fixed by `dsv4-streaming-preamble-fix.diff` in this repo (extends upstream [#31786](https://github.com/sgl-project/sglang/pull/31786) to every DSML marker form): **0/8 cut** |
| 2 | **flashinfer must be exactly 0.6.15.post1** on this base. 0.6.16.post1 segfaults v0.5.16 CUDA-graph capture on SM120 (tested). | pinned in the Dockerfile |
| 3 | **Grammar-constrained decoding is unsupported under DSPARK**: `tool_choice: required` / strict `json_schema` return HTTP 400. | use `tool_choice: auto` (unaffected). Upstream is adding grammar+DSPARK on `main` (#31753) |
| 4 | **`repetition_penalty` is silently ignored under DSPARK verify**, even with #33531 applied — that PR restores only the *additive* penalties; the multiplicative penalizer rides a dense-fallback branch that overlap mode skips. Verified: `repetition_penalty 1.3` stops a repetition probe with spec off, does nothing with spec on. | use `frequency_penalty` / `presence_penalty` — dose-responsive under verify with #33531 applied (0.25–0.3 works; normal prompts stay coherent) |
| 5 | **MoE runner must be `flashinfer_mxfp4`** for this checkpoint on SM120: default triton runner crashes ("Hidden size mismatch"), `flashinfer_trtllm` corrupts FP4 output (sglang#26324). | pinned in the compose |
| 6 | **Custom all-reduce fails CUDA-graph capture** on PCIe GPU pairs (sglang#11957 class). | `--disable-custom-all-reduce` |

Levers we tried and rejected, so you don't re-try them on this hardware: `--enable-flashinfer-allreduce-fusion` (SM90/SM10x only, crashes), `--enable-quant-communications` (NPU only), `--enable-torch-symm-mem` ("Device capability 12 not supported"), NCCL P2P via IOMMU (engages but net-negative for decode latency here), `SGLANG_RAGGED_VERIFY_MODE=compact` (pre-#32467 it IMAs at graph capture; with #32467 it captures, but the depth-gated corruption above is independent of verify mode, so this recipe serves `static`).

## The draft-depth corruption boundary (issue 1, in full)

Spec decode is supposed to be distribution-preserving, so any temp-0 divergence between spec-on and spec-off is a verify-path mis-commit, not a sampling artifact. What we measured, all on this hardware:

| arm | base | γ | verify rows | verdict |
|---|---|---|---|---|
| this stack | v0.5.16 | 3 | 4 | **clean** — 0 events / 10,885 req (5 cold launches) |
| **this stack (shipped)** | v0.5.16 | **4** | **5** | **clean — 0 events / 11,040 req (5 cold launches)** |
| this stack | v0.5.16 | 5 (inferred) | 6 | **dirty** — 10 events / 2,148 req |
| this stack | v0.5.16 | 5 (explicit) | 6 | **dirty** — 7 events / 2,138 req |
| this stack | v0.5.16 | 6 | 7 | **clean** — 0 events / 2,282 req |
| this stack | v0.5.16 | 7 | 8 | **clean** — 0 events / 2,198 req |
| + #32277 clamp backport | v0.5.16 | 5 | 6 | dirty 5/5 |
| upstream `main` @ 211ee642 + rebased stack | main 2026-08-05 | 5 | 6 | **dirty 5/5** — live upstream, not an artifact of this backport stack |

The γ=3 and γ=4 rows are 5-cold-launch arms; γ=5/6/7 are single-launch screens of the same 4-phase load (~2.1–2.3K req), where the γ=5 control produced 10 events. A zero there is a strong signal, not a small-sample artifact.

γ=5 is the checkpoint's native block size and its immediate neighbours are clean, so the most likely explanation is a native-block-size specialised path (kernel instantiation or planner fast path selected when the requested block size equals the head's trained size) being wrong on SM120. Not a capacity, window, or ring bound. We ruled those out by measurement, not argument: c4 compress-state ring capacity (doubling it changed nothing, 32 events / 10,209 req), verify window vs compress ratio (γ=4 and 6 cross the same boundaries and are clean), NaN logits, KV-store padding, the #32277 sentinel clamp, CUTLASS-vs-Triton attention dispatch, and the config inference path (explicit γ=5 is equally dirty).

Supporting evidence:

- Greedy-invariant repro (~10 min, no benchmarks): boot with and without `--speculative-*` at the same shape, send identical prompts at `temperature: 0`, diff. At corrupting shapes the spec-on output is garbage while spec-off is flawless.
- [sglang#32183](https://github.com/sgl-project/sglang/pull/32183) (verifier state rewrite window, `kMaxMTPDraftTokens=4` vs γ+1=6 verify rows) is the closest upstream fix and is necessary but insufficient: with the backport applied and instrumented, `verify_width=6` demonstrably reaches the patched plan kernel, and γ=5 still corrupts. The ≤4-row assumption is broken somewhere one level deeper.
- External corroboration: [sglang#32666](https://github.com/sgl-project/sglang/issues/32666) reports the same boundary on a different backend: depth 5, 23/24 runs corrupt; depth 3, 0/607.
- A NaN tripwire (sanitize-nan-logits ported into the DSPARK verify path) fired zero times across all corrupting launches. The poisoned logits are finite; output classification, not NaN scrubbing, is the right detector.

### Repro protocol (and why easier tests lie)

Two things that do not reproduce this bug:

- **Single-launch test batteries.** The fault is stochastic per server launch. A config can pass a full correctness battery on one boot and corrupt on the next. Anything less than ~5 cold launches per arm is noise.
- **Steady-concurrency soak.** Two 45-minute steady-load soaks (~8K requests) came back clean on a config that then failed 5/5 under bursty load. High accept rate with clean output also occurs, so accept-rate drift is not a usable signature either.

What does reproduce it, 5/5: per arm, 5 cold launches × ~12 min of bursty load each (phases of 3 min @ 4 workers, 3 min @ 20, 2 min @ 2, 4 min @ 14) of streaming chat completions with thinking enabled (`chat_template_kwargs: {thinking: true}`, high reasoning effort), ~20-tool schemas, multi-turn conversations. Score outputs with a salad classifier: U+FFFD replacement chars, punctuation density, single-char spray, trigram loops. The batch-shape sweeps from burst transitions + streaming + thinking-on are what single-shape steady load never exercises.

## Patches

All patches apply to `lmsysorg/sglang:v0.5.16-cu130-runtime`. `prNNNNN-notest.diff` files are unmerged-at-pin-time upstream PR diffs, authorship belongs to their respective PR authors; test hunks stripped where the runtime image lacks the test tree. Paths are rewritten where v0.5.16 predates upstream's kernel-tree restructuring (`kernels/jit/csrc` ↔ `jit_kernel/csrc` etc.); details in each diff header.

| patch | upstream | what it fixes |
|---|---|---|
| `dspark-sm120-decode-dispatch.diff` | ours — [sglang#33407](https://github.com/sgl-project/sglang/pull/33407) | DSPARK crashes at warmup on SM120: draft/verify batches with non-bucket indexer topk (192) fall through to the prefill kernel's `num_tokens > 64` assert. Pads indices to the next instantiated CUTLASS decode bucket with `-1` skip sentinels (scan capped via `topk_length`), Triton fallback for shapes that remain non-dispatchable. Same failure family as [#33134](https://github.com/sgl-project/sglang/issues/33134) (DGX Spark, sm_121) |
| `dsv4-streaming-preamble-fix.diff` | ours — [sglang#33813](https://github.com/sgl-project/sglang/pull/33813), extending [#31786](https://github.com/sgl-project/sglang/pull/31786) | streaming detector returned no normal text on the tool-call branch, discarding assistant prose that shared a delta with the DSML opener. Upstream's fix covers `<｜DSML｜tool_calls>`; V4 also opens with a bare `<｜DSML｜invoke`, so this covers every marker form plus trailing partial-tag prefixes. Fixes known issue 1b |
| `dsv4-c4-ring-depth-scale.diff` | ours — not yet filed | makes the c4 compress-state ring honour draft depth (it was a constant 16 while the SWA ring already scales with `speculative_num_draft_tokens`). Hardening only — measured **not** to fix issue 1; kept because a depth-independent ring is wrong on its face |
| `dsv4-store-padding-guard.diff` | ours — [sglang#33816](https://github.com/sgl-project/sglang/pull/33816) | `store.cuh` fused KV-store kernels skip padded slot-0 writes (implements the kernel's own documented intent) and fix a latent negative-page OOB **write** when the SWA LUT yields `-1`. Memory-safety hardening; verified not the cause of issue 1 |
| `pr32332-notest.diff` | [#32332](https://github.com/sgl-project/sglang/pull/32332) OPEN | DSML streaming-detector buffer poisoning → garbled/duplicated/truncated tool-call arguments + error flood behind OpenAI-compatible proxies |
| `pr32167-notest.diff` | [#32167](https://github.com/sgl-project/sglang/pull/32167) OPEN | tool calls completing in the final streamed delta delivered with empty arguments (common with spec decode) |
| `pr31009-notest.diff` | [#31009](https://github.com/sgl-project/sglang/pull/31009) OPEN | think/tool structural tokens split across stream chunk boundaries leaking into content |
| `pr32320-notest.diff` | [#32320](https://github.com/sgl-project/sglang/pull/32320) MERGED | FlashMLA SM120 page-split rewrote the ENTIRE KV pool every decode step; touched-page mask only copies pages referenced this step. Biggest single decode win in this stack (conc32 +26% measured on an earlier non-DP variant) |
| `pr33098-notest.diff` | [#33098](https://github.com/sgl-project/sglang/pull/33098) MERGED | DSpark draft ForwardBatch misses `original_global_num_tokens_cpu` + non-padded token counts under `--enable-dp-attention` → scheduler crash on first request. **Mandatory for DP+DSPARK** |
| `pr32467-notest.diff` | [#32467](https://github.com/sgl-project/sglang/pull/32467) OPEN | missing `__syncthreads()` race in `plan_compress_prefill_kernel0` → ragged extend misdetected as MTP-uniform → `ragged_id` overrun → illegal memory access at decode-graph capture (shape-dependent) |
| `pr32183-notest.diff` | [#32183](https://github.com/sgl-project/sglang/pull/32183) OPEN | DSpark verifier compressed-KV rewrite window capped at 4 draft tokens regardless of verify width → stale state in verify rows. Necessary but **not sufficient** for issue 1 (see above) |
| `pr32183-hipradix-completion.diff` | ours — v0.5.16-specific, no upstream counterpart | upstream #32183 threads `verify_width` only through `deepseek_v4_backend.py`, but the live raw-verify path on v0.5.16 is the hip_radix backend variant, where it still defaulted to 0 and the write plan kept dropping verifier rows at depth ≥ 4. Without this the row above never executes on the serving path |
| `pr33531-notest.diff` | [#33531](https://github.com/sgl-project/sglang/pull/33531) OPEN | verify path reads a renamed-away attr → ALL sampling penalties silently dead under spec decode on stock v0.5.16. Restores the additive ones (frequency/presence/min_new_tokens); hunks re-derived for v0.5.16. The multiplicative gap remains (known issue 4) |
| `pr32277-notest.diff` | [#32277](https://github.com/sgl-project/sglang/pull/32277) MERGED | clamps all-sentinel DSpark draft rows to token 0 in `_online_combine_kernel` (they otherwise emit an out-of-range token id into verify), plus NaN guards in the reject-sampling path. Measured **not** to fix known issue 1, kept as correctness hardening |
| `pr31017-notest.diff` | [#31017](https://github.com/sgl-project/sglang/pull/31017) MERGED | MoE top-k renormalization epsilon for degenerate rows |
| `pr29927-notest.diff` | [#29927](https://github.com/sgl-project/sglang/pull/29927) OPEN | the SM120 prefill stack: batched sparse-MLA prefill for >64-row calls, chunked indexer metadata, u64-lane page-split, HC-prenorm dispatch, and the env switches the compose file enables (which also need the Dockerfile's sgl-deep-gemm bump). This backport gates the >64-row route on `topk == 512`: ungated, DSPARK's draft indexer (topk=192) lands in the prefill kernel and crash-loops the boot — upstream can't see this because `main` can't boot DSPARK on SM120 without #33407. Details in the diff header |

The two patches from the 2026-08-06 update are ahead of the published image tags; everything else is exactly the image the numbers above were measured on. Retire each backport when it lands in an sglang release. The stack also rebases cleanly onto upstream `main` (three of the diffs drop as merged, the flashinfer pin becomes upstream's own), but note that `main` does not fix known issue 1.

## What here generalizes

The name promises one exact rig. These parts don't need it:

- **The draft-depth corruption (issue 1)** is a property of the DSpark draft head, not of this card count or this engine. If you run DSpark anywhere, vLLM included, and your stack infers the draft depth from the checkpoint, you are at depth 5, the one value we measured as broken. Independent reports match on other hardware ([sglang#32666](https://github.com/sgl-project/sglang/issues/32666), and the same phenotype under vLLM on the same GPUs in [rtx6kpro#53](https://github.com/local-inference-lab/rtx6kpro/issues/53)). Test your own depth before trusting any of us.
- **The patches** are per-bug, not per-rig. The SM120 dispatch fix, the streaming-detector fixes, and the store guard apply to 2× and 8× configs the same way; ormandj's 2×-card repo already carries overlapping fixes.
- **The pins** (flashinfer 0.6.15.post1 on this base; the DeepGEMM constraints) hold for Blackwell workstation silicon generally, not just four of them.
- **What doesn't travel:** every tuned number. TP4/DP4/EP4, mem-fraction, chunk size, γ=4, and all benchmarks are 4×-specific measurements. On different hardware, re-measure; the repro protocol above is the part you can reuse.

## Related work

- [ormandj/sglang-deepseek-v4-flash-sm120](https://github.com/ormandj/sglang-deepseek-v4-flash-sm120) — same model/framework/silicon on 2× RTX PRO 6000, TP=2. The lock-and-verify repo structure there is worth copying.
- [0xSero/deepseek-v4-flash-sm120](https://github.com/0xSero/deepseek-v4-flash-sm120) — the earliest SM120 recipe for this model family.
- [vllm#41834](https://github.com/vllm-project/vllm/pull/41834) — the SM12x/DSpark track on vLLM; also the source of the healthy accept-rate calibration (draft accept 0.1–0.3 is normal for this checkpoint — low accept is not by itself a corruption signal).

## License

Apache-2.0 (see `LICENSE`, `NOTICE`). Patches under `patches/` are derivative works of [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0); the unmerged PR diffs remain the work of their upstream authors.
