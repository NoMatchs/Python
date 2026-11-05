import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x**2 + 3 * x
z = y.sum()

z.backward()  # 自动求导
print(x.grad)  # dz/dx = 2x + 3 = [7, 9]
