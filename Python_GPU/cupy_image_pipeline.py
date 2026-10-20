"""
CuPy 图像处理管线：高斯模糊 → Sobel 边缘检测
自动在脚本所在目录找图片，不需要传参数
"""

import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cnd
from scipy import ndimage as nd
from PIL import Image
import time
import os


# ============================================================
# CPU 版管线
# ============================================================
def cpu_pipeline(img):
    blur = nd.gaussian_filter(img, sigma=2)
    gx = nd.sobel(blur, axis=1)
    gy = nd.sobel(blur, axis=0)
    edge = np.hypot(gx, gy)
    edge = edge / (edge.max() + 1e-8) * 255
    return edge.astype(np.uint8)


# ============================================================
# GPU 版管线
# ============================================================
def gpu_pipeline(d_img):
    blur = cnd.gaussian_filter(d_img, sigma=2)
    gx = cnd.sobel(blur, axis=1)
    gy = cnd.sobel(blur, axis=0)
    edge = cp.hypot(gx, gy)
    edge = edge / (edge.max() + 1e-8) * 255
    return edge.astype(cp.uint8)


# ============================================================
# 计时
# ============================================================
def bench_cpu(fn, *args, n_repeat=5):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) / n_repeat * 1000


def bench_gpu(fn, *args, n_repeat=5):
    fn(*args)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn(*args)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / n_repeat * 1000


# ============================================================
# 自动找图片
# ============================================================
def find_image():
    """在脚本所在目录找第一张图片"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # 优先找 mountain.png
    preferred = os.path.join(script_dir, "mountain.png")
    if os.path.exists(preferred):
        return preferred

    # 否则找目录下第一张图片
    for f in os.listdir(script_dir):
        if f.lower().endswith(exts):
            return os.path.join(script_dir, f)

    return None


# ============================================================
# 主流程
# ============================================================
def main(image_path):
    img_dir = os.path.dirname(os.path.abspath(image_path))
    base = os.path.splitext(os.path.basename(image_path))[0]
    gpu_out = os.path.join(img_dir, f"{base}_edge_gpu.png")
    cpu_out = os.path.join(img_dir, f"{base}_edge_cpu.png")

    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"输入图片: {image_path}")
    print("=" * 60)

    # ---------- 读图 ----------
    t0 = time.perf_counter()
    img = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    t1 = time.perf_counter()
    print(f"图片尺寸: {img.shape[1]} x {img.shape[0]}")
    print(f"读图 + 转灰度: {(t1 - t0) * 1000:.2f} ms")

    # ---------- CPU ----------
    t_cpu = bench_cpu(cpu_pipeline, img)
    print(f"\nCPU (scipy) 管线: {t_cpu:.2f} ms")

    # ---------- GPU 仅计算 ----------
    d_img = cp.asarray(img)
    cp.cuda.Stream.null.synchronize()
    t_gpu_kernel = bench_gpu(gpu_pipeline, d_img)
    print(f"GPU (cupyx) 管线(仅计算): {t_gpu_kernel:.2f} ms")
    print(f"加速比(仅计算): {t_cpu / t_gpu_kernel:.2f}x")

    # ---------- GPU 含传输 ----------
    def gpu_full():
        d = cp.asarray(img)
        out = gpu_pipeline(d)
        return cp.asnumpy(out)

    t_gpu_full = bench_gpu(gpu_full)
    print(f"\nGPU 管线(含 CPU↔GPU 传输): {t_gpu_full:.2f} ms")
    print(f"加速比(含传输): {t_cpu / t_gpu_full:.2f}x")

    # ---------- 结果验证 ----------
    cpu_result = cpu_pipeline(img)
    gpu_result = cp.asnumpy(gpu_pipeline(cp.asarray(img)))

    diff = np.abs(cpu_result.astype(np.int32) - gpu_result.astype(np.int32))
    print(f"\nCPU vs GPU 结果差异:")
    print(f"  最大像素差: {diff.max()}")
    print(f"  平均像素差: {diff.mean():.4f}")
    print(f"  完全一致: {np.array_equal(cpu_result, gpu_result)}")

    # ---------- 保存 ----------
    Image.fromarray(gpu_result).save(gpu_out)
    Image.fromarray(cpu_result).save(cpu_out)
    print(f"\nGPU 结果: {gpu_out}")
    print(f"CPU 结果: {cpu_out}")


if __name__ == "__main__":
    img = find_image()
    if img is None:
        print("脚本目录下没找到图片！请放一张 png/jpg 到:")
        print(os.path.dirname(os.path.abspath(__file__)))
    else:
        main(img)
