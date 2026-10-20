import cupy as cp
import numpy as np
import time

N = 1024
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

A_gpu = cp.asarray(A)
B_gpu = cp.asarray(B)

# 预热
_ = A_gpu @ B_gpu
cp.cuda.Stream.null.synchronize()

# 多次平均
t0 = time.perf_counter()
for _ in range(50):
    C_gpu = A_gpu @ B_gpu
cp.cuda.Stream.null.synchronize()
t1 = time.perf_counter()

t_avg = (t1 - t0) / 50
gflops = 2 * N**3 / t_avg / 1e9
print(f"CuPy (cuBLAS): {t_avg * 1000:.3f} ms, {gflops:.0f} GFLOP/s")
