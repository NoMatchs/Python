import math
import torch
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# 1. 位置编码生成函数 (与之前代码一致)
# ==========================================
def get_positional_encoding(d_model: int, max_len: int = 5000):
    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings


# ==========================================
# 2. 可视化函数
# ==========================================
def visualize_positional_encoding(d_model=128, max_len=100):
    pe = (
        get_positional_encoding(d_model, max_len).squeeze(1).numpy()
    )  # [max_len, d_model]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 视图 1：热力图（论文原图风格） ---
    # 每一行是一个位置，每一列是一个维度
    im1 = axes[0].imshow(pe, cmap="viridis", aspect="auto")
    axes[0].set_title(
        f"Positional Encoding Heatmap\n(max_len={max_len}, d_model={d_model})",
        fontsize=13,
    )
    axes[0].set_xlabel("Embedding Dimension (d_model)")
    axes[0].set_ylabel("Position (Sequence Index)")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # --- 视图 2：几个特定维度的波浪曲线 ---
    # 取第 0, 1, 4, 5, 20, 21 维展示
    dims_to_plot = [0, 1, 4, 5, 20, 21]
    for dim in dims_to_plot:
        axes[1].plot(pe[:max_len, dim], label=f"dim {dim}", linewidth=1.5)

    axes[1].set_title("Positional Encoding Waves (Select Dimensions)", fontsize=13)
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Value")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("positional_encoding.png", dpi=150)
    plt.show()

    # --- 视图 3：位置之间的余弦相似度矩阵（可选，展示位置相对关系） ---
    print("生成位置相似度矩阵中...")
    # 对所有位置向量做 L2 归一化，然后计算两两余弦相似度
    pe_norm = pe / np.linalg.norm(pe, axis=1, keepdims=True)
    similarity = pe_norm @ pe_norm.T

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im2 = ax2.imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_title("Cosine Similarity Between Positions", fontsize=13)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Position")
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("position_similarity.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # 参数配置：dim 可以调小一点方便看清每个维度的变化
    visualize_positional_encoding(d_model=128, max_len=100)
    print("可视化完成！图片已保存为 positional_encoding.png 和 position_similarity.png")
