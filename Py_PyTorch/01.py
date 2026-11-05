import torch
import time

device = torch.device("cuda")

# 大张量，在 GPU 上做前向 + 反向
x = torch.randn(1000, 1000, device=device, requires_grad=True)

t0 = time.perf_counter()
for _ in range(100):
    y = (x**2).sum()
    y.backward()
    x.grad.zero_()  # 清空梯度，否则会累加
torch.cuda.synchronize()
t1 = time.perf_counter()

print(f"GPU 前向+反向 100 次: {(t1 - t0) * 1000:.1f} ms")
print(f"梯度形状: {x.grad.shape}")
print(f"梯度是否在 GPU: {x.grad.is_cuda}")
