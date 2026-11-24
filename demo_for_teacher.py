import numpy as np
import cupy as cp
import time

# 1. 降低精度到 float32，这是科学计算和 AI 的常用精度，不仅省显存，而且快得多
N = 20000
print(f"正在生成 {N}x{N} 的随机矩阵 (Float32)...")

A_cpu = np.random.rand(N, N).astype(np.float32)
B_cpu = np.random.rand(N, N).astype(np.float32)

# ==========================================
# 第一回合：CPU (NumPy)
# ==========================================
print("\n[1/2] 正在用 CPU 计算...")
t0 = time.time()
C_cpu = np.dot(A_cpu, B_cpu)
t_cpu = time.time() - t0
print(f"CPU 纯计算耗时: {t_cpu:.3f} 秒")

# ==========================================
# 第二回合：GPU (CuPy)
# ==========================================
print("\n[2/2] 正在用 GPU 计算...")
# 先把数据传给显卡（预热，不计入时间）
A_gpu = cp.asarray(A_cpu)
B_gpu = cp.asarray(B_cpu)
cp.cuda.Stream.null.synchronize()  # 等传输完

# 开始计时：只算纯计算时间
t0 = time.time()
C_gpu = cp.dot(A_gpu, B_gpu)
cp.cuda.Stream.null.synchronize()
t_gpu = time.time() - t0
print(f"GPU 纯计算耗时: {t_gpu:.3f} 秒")

# ==========================================
# 结果对比
# ==========================================
print(f"\n🚀 显卡纯计算加速了约 {t_cpu / t_gpu:.1f} 倍！")

# 确认结果一致
error = cp.max(cp.abs(C_gpu - cp.asarray(C_cpu)))
print(f"结果误差检查: {error:.6f} (应该非常接近0)")
