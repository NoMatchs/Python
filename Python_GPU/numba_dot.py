import numpy as np
from numba import cuda
import time


# ---------- 归约 kernel ----------
@cuda.jit
def dot_partial_kernel(a, b, partial):
    # shared memory 用 float64 做累加
    s = cuda.shared.array(shape=0, dtype=np.float64)

    tx = cuda.threadIdx.x
    i = cuda.grid(1)

    if i < a.size:
        # a, b 是什么类型，这里就按什么类型读，再转 float64 累加
        s[tx] = np.float64(a[i]) * np.float64(b[i])
    else:
        s[tx] = 0.0

    cuda.syncthreads()

    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tx < stride:
            s[tx] += s[tx + stride]
        cuda.syncthreads()
        stride //= 2

    if tx == 0:
        partial[cuda.blockIdx.x] = s[0]


def bench_pcie_bandwidth(arr):
    """单独测一次 CPU->GPU 拷贝带宽"""
    # 预热
    _ = cuda.to_device(arr[:1000])
    cuda.synchronize()

    t0 = time.perf_counter()
    d = cuda.to_device(arr)
    cuda.synchronize()
    t1 = time.perf_counter()

    size_mb = arr.nbytes / (1024 * 1024)
    ms = (t1 - t0) * 1000
    bw = size_mb / (t1 - t0) / 1024  # GB/s
    print(f"  拷贝 {size_mb:.0f} MB: {ms:.2f} ms, 带宽 {bw:.2f} GB/s")
    return d


def gpu_dot(d_a, d_b, threads_per_block=256, n_repeat=10):
    """只测 kernel，数据已经在 GPU 上"""
    N = d_a.size
    blocks = (N + threads_per_block - 1) // threads_per_block
    shared_bytes = threads_per_block * 8  # float64

    d_partial = cuda.device_array(blocks, dtype=np.float64)

    # 预热
    dot_partial_kernel[blocks, threads_per_block, 0, shared_bytes](d_a, d_b, d_partial)
    cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        dot_partial_kernel[blocks, threads_per_block, 0, shared_bytes](
            d_a, d_b, d_partial
        )
    cuda.synchronize()
    t_kernel = (time.perf_counter() - t0) * 1000 / n_repeat

    result = d_partial.copy_to_host().sum()
    return result, t_kernel


def main():
    N = 100_000_000
    print(f"N = {N:,}")
    print("=" * 60)

    # ========================================================
    # 实验 A：输入用 float32（复现你之前的结果）
    # ========================================================
    print("\n【实验 A】输入 float32")
    a32 = np.random.rand(N).astype(np.float32)
    b32 = np.random.rand(N).astype(np.float32)

    np.dot(a32, b32)  # 预热
    t0 = time.perf_counter()
    cpu32 = np.dot(a32, b32)
    t1 = time.perf_counter()
    t_cpu32 = (t1 - t0) * 1000
    print(f"  CPU: {t_cpu32:.2f} ms, 结果 {cpu32:.2f}")

    print("  --- PCIe 带宽 ---")
    d_a32 = bench_pcie_bandwidth(a32)
    d_b32 = bench_pcie_bandwidth(b32)

    gpu32, t_k32 = gpu_dot(d_a32, d_b32)
    print(f"  GPU 仅 kernel: {t_k32:.2f} ms, 结果 {gpu32:.2f}")
    rel32 = abs(cpu32 - gpu32) / abs(cpu32)
    print(f"  相对误差: {rel32:.3e}")

    # 释放显存
    del d_a32, d_b32
    cuda.current_context().deallocations.clear()

    # ========================================================
    # 实验 B：输入用 float64（验证精度）
    # ========================================================
    print("\n" + "=" * 60)
    print("【实验 B】输入 float64")
    a64 = a32.astype(np.float64)
    b64 = b32.astype(np.float64)

    np.dot(a64, b64)  # 预热
    t0 = time.perf_counter()
    cpu64 = np.dot(a64, b64)
    t1 = time.perf_counter()
    t_cpu64 = (t1 - t0) * 1000
    print(f"  CPU: {t_cpu64:.2f} ms, 结果 {cpu64:.2f}")

    print("  --- PCIe 带宽 ---")
    d_a64 = bench_pcie_bandwidth(a64)
    d_b64 = bench_pcie_bandwidth(b64)

    gpu64, t_k64 = gpu_dot(d_a64, d_b64)
    print(f"  GPU 仅 kernel: {t_k64:.2f} ms, 结果 {gpu64:.2f}")
    rel64 = abs(cpu64 - gpu64) / abs(cpu64)
    print(f"  相对误差: {rel64:.3e}")

    # ========================================================
    # 汇总
    # ========================================================
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(
        f"{'':20}{'CPU(ms)':>10}{'GPU kernel(ms)':>16}{'kernel加速':>12}{'相对误差':>12}"
    )
    print(
        f"{'float32 输入':20}{t_cpu32:>10.2f}{t_k32:>16.2f}{t_cpu32 / t_k32:>11.2f}x{rel32:>12.3e}"
    )
    print(
        f"{'float64 输入':20}{t_cpu64:>10.2f}{t_k64:>16.2f}{t_cpu64 / t_k64:>11.2f}x{rel64:>12.3e}"
    )


if __name__ == "__main__":
    main()
