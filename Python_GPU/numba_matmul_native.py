import numpy as np
from numba import cuda
import time


@cuda.jit
def matmul_naive(A, B, C):
    # cuda.grid(2) 直接给二维全局坐标
    row, col = cuda.grid(2)

    if row < C.shape[0] and col < C.shape[1]:
        s = 0.0
        for k in range(A.shape[1]):
            s += A[row, k] * B[k, col]
        C[row, col] = s


def gpu_matmul_naive(A, B, threads_per_block=(16, 16), n_repeat=10):
    """只测 kernel，数据已经在 GPU 上"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((M, N), dtype=np.float32)

    blocks = (
        (M + threads_per_block[0] - 1) // threads_per_block[0],
        (N + threads_per_block[1] - 1) // threads_per_block[1],
    )

    # 预热
    matmul_naive[blocks, threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        matmul_naive[blocks, threads_per_block](d_A, d_B, d_C)
    cuda.synchronize()
    t1 = time.perf_counter()
    t_kernel = (t1 - t0) * 1000 / n_repeat

    return d_C.copy_to_host(), t_kernel


def main():
    N = 1024
    print(f"矩阵规模: {N} x {N}")
    print("=" * 60)

    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # ---------- CPU 基线 ----------
    np.dot(A, B)  # 预热
    t0 = time.perf_counter()
    C_cpu = A @ B
    t1 = time.perf_counter()
    t_cpu = (t1 - t0) * 1000
    print(f"NumPy (CPU, 多线程 BLAS): {t_cpu:.3f} ms")

    # ---------- GPU naive ----------
    C_gpu, t_gpu = gpu_matmul_naive(A, B)
    print(f"GPU naive kernel:         {t_gpu:.3f} ms")
    print(f"加速比 (CPU / GPU):       {t_cpu / t_gpu:.2f}x")

    # ---------- 验证 ----------
    rel_err = np.max(np.abs(C_cpu - C_gpu)) / np.max(np.abs(C_cpu))
    print(f"\n最大相对误差: {rel_err:.3e}")
    print(f"结果是否正确: {rel_err < 1e-4}")

    # ---------- 算一下理论性能 ----------
    flops = 2 * N**3
    gflops_cpu = flops / (t_cpu / 1000) / 1e9
    gflops_gpu = flops / (t_gpu / 1000) / 1e9
    print(f"\nCPU 算力: {gflops_cpu:.2f} GFLOP/s")
    print(f"GPU 算力: {gflops_gpu:.2f} GFLOP/s")
    print(f"(RTX 4060 Laptop FP32 峰值约 15000 GFLOP/s)")


if __name__ == "__main__":
    main()
