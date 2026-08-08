# SGLang v0.5.16 + the patch set required to serve DeepSeek-V4-Flash-0731
# on 4x RTX PRO 6000 Blackwell (SM120) at TP4/DP4/EP4 with DSPARK
# speculative decoding. See README.md for what each patch fixes and why.
#
# Patch order matters:
#   - pr32320 touches the same file/function as dspark-sm120-decode-dispatch
#     and must apply AFTER it (hunks verified non-overlapping in that order).
#   - pr33531 and later diffs touch dspark_verify.py; order verified on a
#     clean v0.5.16 worktree.
#   - pr29927 was generated against the full preceding stack (it shares
#     files with flash_mla_sm120 and server_args hunks earlier in the
#     list); the nine diffs after it were generated against the stack up
#     to and including it.
#   - pr30096-family (consolidation of merged upstream #30096 + #32393 +
#     #32409, re-anchored to v0.5.16) touches spec_utils.py,
#     dspark_verify.py, dflash_utils.py and scheduler.py after pr33531,
#     pr32880 and pr33805 have already modified them, so it applies LAST.
# All prNNNNN-notest.diff files are upstream PR diffs pinned at review time
# with test-file hunks stripped (the runtime image ships no test/ tree);
# pr31017 retains its test hunk because the path exists in-image.
#
# Build: docker build -t sglang:v0.5.16-dsv4-sm120 .
FROM lmsysorg/sglang:v0.5.16-cu130-runtime

COPY patches/*.diff /tmp/patches/
RUN cd /sgl-workspace/sglang \
    && git apply --verbose /tmp/patches/pr32332-notest.diff \
    && git apply --verbose /tmp/patches/pr32167-notest.diff \
    && git apply --verbose /tmp/patches/pr31009-notest.diff \
    && git apply --verbose /tmp/patches/dspark-sm120-decode-dispatch.diff \
    && git apply --verbose /tmp/patches/pr32320-notest.diff \
    && git apply --verbose /tmp/patches/pr33098-notest.diff \
    && git apply --verbose /tmp/patches/pr32467-notest.diff \
    && git apply --verbose /tmp/patches/pr32183-notest.diff \
    && git apply --verbose /tmp/patches/pr33531-notest.diff \
    && git apply --verbose /tmp/patches/dsv4-store-padding-guard.diff \
    && git apply --verbose /tmp/patches/pr31017-notest.diff \
    && git apply --verbose /tmp/patches/pr32277-notest.diff \
    && git apply --verbose /tmp/patches/pr32183-hipradix-completion.diff \
    && git apply --verbose /tmp/patches/dsv4-streaming-preamble-fix.diff \
    && git apply --verbose /tmp/patches/dsv4-c4-ring-depth-scale.diff \
    && git apply --verbose /tmp/patches/pr29927-notest.diff \
    && git apply --verbose /tmp/patches/pr31170-notest.diff \
    && git apply --verbose /tmp/patches/pr31835-notest.diff \
    && git apply --verbose /tmp/patches/pr32880-notest.diff \
    && git apply --verbose /tmp/patches/pr27199-notest.diff \
    && git apply --verbose /tmp/patches/dsv4-compact-gamma-runtime.diff \
    && git apply --verbose /tmp/patches/pr33805-notest.diff \
    && git apply --verbose /tmp/patches/pr33795-notest.diff \
    && git apply --verbose /tmp/patches/pr32379-notest.diff \
    && git apply --verbose /tmp/patches/pr32700-notest.diff \
    && git apply --verbose /tmp/patches/pr31975-notest.diff \
    && git apply --verbose /tmp/patches/pr33568-notest.diff \
    && git apply --verbose /tmp/patches/pr32686-notest.diff \
    && git apply --verbose /tmp/patches/pr30096-family-notest.diff \
    && git apply --verbose /tmp/patches/pr33459-notest.diff \
    && git apply --verbose /tmp/patches/dspark-return-logprob.diff \
    && python3 -c "import ast; [ast.parse(open(f).read()) for f in ('python/sglang/srt/function_call/deepseekv32_detector.py','python/sglang/srt/function_call/base_format_detector.py','python/sglang/srt/entrypoints/openai/serving_chat.py','python/sglang/srt/parser/reasoning_parser.py','python/sglang/kernels/ops/attention/flash_mla_sm120.py','python/sglang/srt/speculative/dspark_components/dspark_draft.py','python/sglang/jit_kernel/dsv4/compress.py','python/sglang/srt/layers/attention/deepseek_v4_backend.py','python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py','python/sglang/srt/layers/attention/dsv4/compressor_v2.py','python/sglang/srt/layers/attention/dsv4/indexer.py','python/sglang/srt/layers/attention/dsv4/metadata.py','python/sglang/srt/speculative/dflash_utils.py','python/sglang/srt/speculative/dspark_components/dspark_verify.py','python/sglang/srt/speculative/spec_utils.py','python/sglang/srt/layers/moe/topk.py','python/sglang/kernels/ops/layernorm/mhc.py','python/sglang/srt/layers/deep_gemm_wrapper/configurer.py','python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py','python/sglang/srt/layers/moe/moe_runner/deep_gemm.py','python/sglang/srt/model_loader/utils.py','python/sglang/srt/models/deepseek_v4.py','python/sglang/srt/models/deepseek_v4_dspark.py','python/sglang/srt/server_args.py','python/sglang/srt/managers/data_parallel_controller.py','python/sglang/srt/managers/schedule_policy.py','python/sglang/srt/managers/prefill_delayer.py','python/sglang/srt/managers/scheduler.py','python/sglang/srt/managers/io_struct.py','python/sglang/srt/constrained/base_grammar_backend.py','python/sglang/srt/model_executor/model_runner.py','python/sglang/srt/sampling/sampling_batch_info.py','python/sglang/srt/speculative/dflash_worker_v2.py','python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py','python/sglang/srt/speculative/eagle_info.py','python/sglang/srt/speculative/eagle_utils.py','python/sglang/srt/speculative/eagle_worker_common.py','python/sglang/srt/speculative/ngram_info.py','python/sglang/srt/speculative/ngram_worker.py','python/sglang/srt/speculative/spec_info.py','python/sglang/srt/layers/logprob_processor.py')]; print('all patched files parse OK')"

# flashinfer 0.6.14 (base image) -> 0.6.15.post1 (sglang main's own pin).
# Do NOT bump to 0.6.16.post1: it segfaults v0.5.16 CUDA-graph capture on
# SM120 (tested). All three flashinfer packages must move in lockstep
# (env.py hard-fails on a cubin/python version mismatch); 0.6.15.post1
# cubin + CUDA-suffixed jit-cache wheels ship only as GitHub release
# assets, so install by direct URL. First boot after a flashinfer bump
# recompiles the JIT cache: slow first start expected.
RUN pip install --no-cache-dir \
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.15.post1/flashinfer_python-0.6.15.post1-py3-none-any.whl" \
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.15.post1/flashinfer_cubin-0.6.15.post1-py3-none-any.whl" \
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.15.post1/flashinfer_jit_cache-0.6.15.post1%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl" \
    && python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

# sgl-deep-gemm 0.1.4.post1 (base image) -> 0.1.5.post1. 0.1.4's host lib
# asserts arch 9/10 only; 0.1.5's _C.so admits SM120 in fp8_paged_mqa_logits
# and the hyperconnection tf32 GEMM — the two kernels the #29927 env recipe
# needs (SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0 + SGLANG_OPT_USE_TILELANG_
# INDEXER=0 route the indexer to DeepGEMM; SGLANG_OPT_DEEPGEMM_HC_PRENORM=1
# enables the prenorm path; see the compose file). This is the exact version
# upstream main pins, i.e. what #29927 was developed against. DeepGEMM's FP4
# grouped MoE GEMM remains broken on SM120 in DeepGEMM itself (CUDA 719
# launch failure inside its own warmup sweep at DSv4 per-rank shapes), so
# --moe-runner-backend stays flashinfer_mxfp4 — do not switch it to
# deep_gemm on this hardware. Do NOT bump to 0.1.5.post2: it asserts on
# v0.5.16's weight pipeline (wants main's quant stack). New JIT cache keys,
# so first boot is slow again. flashinfer stays pinned above.
RUN pip install --no-cache-dir sgl-deep-gemm==0.1.5.post1 \
    && python3 -c "import importlib.metadata as md, deep_gemm; v=md.version('sgl-deep-gemm'); assert v=='0.1.5.post1', v; print('sgl-deep-gemm', v)" \
    && python3 -c "import flashinfer; v=flashinfer.__version__; assert v=='0.6.15.post1', v; print('flashinfer still', v)"
