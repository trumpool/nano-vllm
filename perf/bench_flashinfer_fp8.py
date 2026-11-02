"""Can FlashInfer's paged decode attention read an FP8 KV cache on this GPU, and how fast is it vs bf16?

Standalone (torch + flashinfer only). Shapes are one Qwen3-0.6B layer: 16 q heads, 8 kv heads, head_dim 128,
page size 256, NHD layout (same as nano-vllm). Every sequence gets its own pages (no reuse, so no L2 hits).
FP8 is e4m3 with one scale per tensor (K and V separately).

    python perf/bench_flashinfer_fp8.py
"""
import torch
import flashinfer

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE = 16, 8, 128, 256
BATCH_SIZES = [32, 64, 128, 256]
CONTEXT_LENS = [512, 1024, 2048]
PEAK_GBPS = 896


def time_gpu(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3


def to_fp8(x):
    scale = x.abs().amax().float() / torch.finfo(torch.float8_e4m3fn).max
    return (x.float() / scale).to(torch.float8_e4m3fn), scale.item()


torch.manual_seed(0)
workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
print(f"flashinfer {flashinfer.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name()}")
print(f"{'bs':>4s} {'ctx':>5s} | {'kernel':>14s} {'time':>9s} {'BW':>9s} {'%peak':>6s} | {'speedup':>7s} {'max err':>8s}")
for bs in BATCH_SIZES:
    for ctx in CONTEXT_LENS:
        pages_per_seq = ctx // PAGE
        num_pages = bs * pages_per_seq
        k = torch.randn(num_pages, PAGE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        k8, k_scale = to_fp8(k)
        v8, v_scale = to_fp8(v)
        q = torch.randn(bs, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        indices = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
        indptr = torch.arange(0, num_pages + 1, pages_per_seq, device="cuda", dtype=torch.int32)
        last_page_len = torch.full((bs,), PAGE, device="cuda", dtype=torch.int32)

        results = {}
        for name, kv, kv_dtype, scales, tc in (
            ("bf16", (k, v), torch.bfloat16, {}, False),
            ("bf16 tc", (k, v), torch.bfloat16, {}, True),
            ("fp8", (k8, v8), torch.float8_e4m3fn, dict(k_scale=k_scale, v_scale=v_scale), False),
            ("fp8 tc", (k8, v8), torch.float8_e4m3fn, dict(k_scale=k_scale, v_scale=v_scale), True),
        ):
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=tc)
            w.plan(indptr, indices, last_page_len, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE,
                   q_data_type=torch.bfloat16, kv_data_type=kv_dtype)
            out = w.run(q, kv, **scales)
            t = time_gpu(lambda: w.run(q, kv, **scales))
            results[name] = (t, out)

        ref_t, ref_out = results["bf16"]
        for name, (t, out) in results.items():
            nbytes = bs * ctx * 2 * NUM_KV_HEADS * HEAD_DIM * (1 if "fp8" in name else 2)
            bw = nbytes / t / 1e9
            err = (out.float() - ref_out.float()).abs().max().item()
            print(f"{bs:4d} {ctx:5d} | {name:>14s} {t * 1e6:7.1f}us {bw:6.0f}GB/s {bw / PEAK_GBPS:6.0%} | "
                  f"{ref_t / t:6.2f}x {err:8.4f}")
        del k, v, k8, v8
