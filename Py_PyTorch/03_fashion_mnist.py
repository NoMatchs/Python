import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ============================================================
    # 关键改动：训练集用数据增强，测试集只用基础变换
    # ============================================================
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),  # 随机裁剪 + 填充
            transforms.RandomAffine(
                degrees=10,  # 随机旋转 ±10 度
                translate=(0.1, 0.1),
            ),  # 随机平移 10%
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"训练集: {len(train_dataset)} 张, 测试集: {len(test_dataset)} 张")
    print(f"训练增强: RandomCrop(28, padding=4) + RandomAffine(±10°, ±10%)")

    model = CNN().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print("\n开始训练...")
    print("=" * 60)

    for epoch in range(epochs):
        t0 = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        t1 = time.perf_counter()
        print(
            f"Epoch {epoch + 1:2d} | "
            f"train loss {train_loss:.4f} acc {train_acc * 100:.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc * 100:.2f}% | "
            f"耗时 {t1 - t0:.2f}s"
        )

    print("=" * 60)
    print(f"最终测试准确率: {test_acc * 100:.2f}%")
