import torch
import torch.nn as nn

# ============ 1. 准备数据 ============
# 真实参数：w=3, b=2
torch.manual_seed(42)
N = 1000
x = torch.randn(N, 1)
true_w, true_b = 3.0, 2.0
y = true_w * x + true_b + 0.1 * torch.randn(N, 1)  # 加噪声

# 搬到 GPU
device = torch.device("cuda")
x = x.to(device)
y = y.to(device)


# ============ 2. 定义模型 ============
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 输入 1 维，输出 1 维

    def forward(self, x):
        return self.linear(x)


model = LinearRegression().to(device)

# ============ 3. 损失函数 + 优化器 ============
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ============ 4. 训练循环 ============
epochs = 100
for epoch in range(epochs):
    # 前向
    pred = model(x)
    loss = criterion(pred, y)

    # 反向
    # optimizer.zero_grad()  # 清空上一步的梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1:3d}: loss = {loss.item():.6f}")

# ============ 5. 查看学到的参数 ============
w_learned = model.linear.weight.item()
b_learned = model.linear.bias.item()
print(f"\n真实参数: w={true_w}, b={true_b}")
print(f"学到的参数: w={w_learned:.4f}, b={b_learned:.4f}")
