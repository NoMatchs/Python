import cupy as cp
import time

# 1. 在 GPU 上创建数组
x_gpu = cp.arange(10_000, dtype=cp.float32)
y_gpu = cp.ones((10_000, 10_000), dtype=cp.float32)

# 2. 执行矩阵向量乘法 (在 GPU 上运算)
start = time.time()
result_gpu = cp.dot(y_gpu, x_gpu)
cp.cuda.Stream.null.synchronize()  # 等待 GPU 运算完成
end = time.time()

print(f"GPU 运算耗时: {(end - start) * 1000:.2f} ms")
print(f"结果类型: {type(result_gpu)}")
print(f"前 5 个元素: {result_gpu[:5]}")

# 3. 需要时，将结果传回 CPU (NumPy 数组)
result_cpu = cp.asnumpy(result_gpu)
print(f"传回 CPU 后的类型: {type(result_cpu)}")
