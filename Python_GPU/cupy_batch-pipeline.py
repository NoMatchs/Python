"""
批量图像处理：逐张 vs 批量 vs 批量+Stream
对比三种方式的吞吐量（张/秒）
"""

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cnd
from PIL import Image
import time
import os


# ============================================================
# 生成测试图
# ============================================================
def generate_test_images(n, size=512, out_dir="batch_test"):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    rng = np.random.default_rng(42)

    for i in range(n):
        img = np.zeros((size, size), dtype=np.uint8)
        for _ in range(5):
            x1, y1 = rng.integers(0, size - 100, 2)
            w, h = rng.integers(50, 150, 2)
            img[y1 : y1 + h, x1 : x1 + w] = rng.integers(100, 255)
        path = os.path.join(out_dir, f"test_{i:03d}.png")
        Image.fromarray(img).save(path)
        paths.append(path)
    return paths


# ============================================================
# 图像处理管线（GPU）
# 注意：sobel 对 3D 数组语义不对，必须逐张循环
# ============================================================
def gpu_pipeline(d_img):
    if d_img.ndim == 3:
        sigma = (0, 2, 2)  # batch 轴不滤波
    else:
        sigma = 2

    blur = cnd.gaussian_filter(d_img, sigma=sigma)

    if blur.ndim == 3:
        N = blur.shape[0]
        # 逐张做 sobel，避免 3D 语义问题
        gx = cp.stack([cnd.sobel(blur[i], axis=-1) for i in range(N)])
        gy = cp.stack([cnd.sobel(blur[i], axis=-2) for i in range(N)])
    else:
        gx = cnd.sobel(blur, axis=-1)
        gy = cnd.sobel(blur, axis=-2)

    return cp.hypot(gx, gy)


# ============================================================
# 方式 A：逐张处理
# ============================================================
def process_one_by_one(paths):
    results = []
    for p in paths:
        img = np.array(Image.open(p).convert("L"), dtype=np.float32)
        d_img = cp.asarray(img)
        d_out = gpu_pipeline(d_img)
        results.append(cp.asnumpy(d_out))
    return results


# ============================================================
# 方式 B：批量处理
# ============================================================
def process_batch(paths):
    imgs = np.stack(
        [np.array(Image.open(p).convert("L"), dtype=np.float32) for p in paths]
    )
    d_imgs = cp.asarray(imgs)
    d_out = gpu_pipeline(d_imgs)
    return cp.asnumpy(d_out)


# ============================================================
# 方式 C：批量 + Stream 并发
# ============================================================
def process_batch_stream(paths, n_streams=4):
    imgs = np.stack(
        [np.array(Image.open(p).convert("L"), dtype=np.float32) for p in paths]
    )
    N = imgs.shape[0]
    chunk = (N + n_streams - 1) // n_streams

    streams = [cp.cuda.Stream() for _ in range(n_streams)]
    results = []

    for i in range(n_streams):
        start = i * chunk
        end = min(start + chunk, N)
        if start >= end:
            continue
        with streams[i]:
            d_chunk = cp.asarray(imgs[start:end])
            d_out = gpu_pipeline(d_chunk)
            results.append(cp.asnumpy(d_out))

    for s in streams:
        s.synchronize()

    return np.concatenate(results, axis=0)


# ============================================================
# 计时
# ============================================================
def bench(fn, *args, n_repeat=1):
    fn(*args)
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn(*args)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / n_repeat * 1000


# ============================================================
# 主流程
# ============================================================
def main():
    N = 100
    SIZE = 512

    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"生成 {N} 张 {SIZE}x{SIZE} 测试图...")
    paths = generate_test_images(N, SIZE)
    print(f"完成，共 {len(paths)} 张")
    print("=" * 60)

    # ---------- 诊断：只比第一张图 ----------
    print("\n诊断：只比第一张图")
    r_a = process_one_by_one(paths[:1])[0]
    r_b = process_batch(paths[:1])[0]
    diff = np.max(np.abs(r_a - r_b))
    print(f"  A vs B 单张最大差异: {diff:.6f}")

    # ---------- 性能 ----------
    t_a = bench(process_one_by_one, paths)
    print(f"\n方式 A（逐张处理）:")
    print(f"  总耗时: {t_a:.2f} ms")
    print(f"  吞吐量: {N / t_a * 1000:.1f} 张/秒")

    t_b = bench(process_batch, paths)
    print(f"\n方式 B（批量处理）:")
    print(f"  总耗时: {t_b:.2f} ms")
    print(f"  吞吐量: {N / t_b * 1000:.1f} 张/秒")

    t_c = bench(process_batch_stream, paths, 4)
    print(f"\n方式 C（批量 + 4 Stream 并发）:")
    print(f"  总耗时: {t_c:.2f} ms")
    print(f"  吞吐量: {N / t_c * 1000:.1f} 张/秒")

    # ---------- 验证一致性 ----------
    print("\n" + "=" * 60)
    print("结果一致性验证（前 5 张）")
    r_a5 = np.stack(process_one_by_one(paths[:5]))
    r_b5 = process_batch(paths[:5])
    r_c5 = process_batch_stream(paths[:5], 4)

    for name, r in [("A", r_a5), ("B", r_b5), ("C", r_c5)]:
        diff = np.max(np.abs(r_b5 - r))
        rel = diff / (np.max(np.abs(r_b5)) + 1e-8)
        print(f"  {name} vs B: 最大绝对差 {diff:.6f}, 相对差 {rel:.3e}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"{'方式':<20}{'耗时(ms)':>12}{'吞吐(张/秒)':>16}{'相对A加速':>12}")
    print(f"{'A 逐张':<20}{t_a:>12.2f}{N / t_a * 1000:>16.1f}{1.0:>11.2f}x")
    print(f"{'B 批量':<20}{t_b:>12.2f}{N / t_b * 1000:>16.1f}{t_a / t_b:>11.2f}x")
    print(
        f"{'C 批量+Stream':<20}{t_c:>12.2f}{N / t_c * 1000:>16.1f}{t_a / t_c:>11.2f}x"
    )


if __name__ == "__main__":
    main()
