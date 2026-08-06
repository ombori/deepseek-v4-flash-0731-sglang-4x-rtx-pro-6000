# SGLang v0.5.16 + the patch set required to serve DeepSeek-V4-Flash-0731
# on 4x RTX PRO 6000 Blackwell (SM120) at TP4/DP4/EP4 with DSPARK
# speculative decoding. See README.md for what each patch fixes and why.
#
# Patch order matters:
#   - pr32320 touches the same file/function as dspark-sm120-decode-dispatch
#     and must apply AFTER it (hunks verified non-overlapping in that order).
#   - pr33531 and later diffs touch dspark_verify.py; order verified on a
#     clean v0.5.16 worktree.
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
    && git apply --verbose /tmp/patches/dsv4-streaming-preamble-fix.diff \
    && git apply --verbose /tmp/patches/dsv4-c4-ring-depth-scale.diff \
    && python3 -c "import ast; [ast.parse(open(f).read()) for f in ('python/sglang/srt/function_call/deepseekv32_detector.py','python/sglang/srt/function_call/base_format_detector.py','python/sglang/srt/entrypoints/openai/serving_chat.py','python/sglang/srt/parser/reasoning_parser.py','python/sglang/kernels/ops/attention/flash_mla_sm120.py','python/sglang/srt/speculative/dspark_components/dspark_draft.py','python/sglang/jit_kernel/dsv4/compress.py','python/sglang/srt/layers/attention/deepseek_v4_backend.py','python/sglang/srt/layers/attention/dsv4/compressor_v2.py','python/sglang/srt/speculative/dflash_utils.py','python/sglang/srt/speculative/dspark_components/dspark_verify.py','python/sglang/srt/speculative/spec_utils.py','python/sglang/srt/layers/moe/topk.py')]; print('all patched files parse OK')"

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
