import cupy as cp
import numpy as np
import time

print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print(f"总显存: {cp.cuda.Device(0).mem_info[1] / 1024**3:.2f} GB")
print("=" * 60)

pool = cp.get_default_memory_pool()
pinned_pool = cp.get_default_pinned_memory_pool()


def show_mem(tag):
    free, total = cp.cuda.Device(0).mem_info
    used = total - free
    print(f"[{tag}]")
    print(f"  设备已用:   {used / 1024**2:8.1f} MB")
    print(f"  池 used:    {pool.used_bytes() / 1024**2:8.1f} MB")
    print(f"  池 total:   {pool.total_bytes() / 1024**2:8.1f} MB")


show_mem("初始")

# ---------- 1. 反复分配/释放，看池子行为 ----------
print("\n--- 反复分配 100 个 10MB 数组 ---")
t0 = time.perf_counter()
for i in range(100):
    a = cp.empty(10 * 1024 * 1024 // 4, dtype=cp.float32)  # 10MB
    del a
t1 = time.perf_counter()
print(f"耗时: {(t1 - t0) * 1000:.1f} ms")
show_mem("循环后")

# ---------- 2. 手动释放池子 ----------
print("\n--- 手动释放池子 ---")
pool.free_all_blocks()
pinned_pool.free_all_blocks()
show_mem("释放后")

# ---------- 3. 大数组逼近显存上限 ----------
print("\n--- 分配一个 2GB 数组 ---")
try:
    big = cp.empty(2 * 1024**3 // 4, dtype=cp.float32)
    show_mem("分配 2GB 后")
    del big
    show_mem("删除 2GB 后（注意池子没还）")
    pool.free_all_blocks()
    show_mem("再次 free_all_blocks 后")
except cp.cuda.memory.OutOfMemoryError as e:
    print(f"OOM: {e}")

# ---------- 4. 用 with 块限定作用域 ----------
print("\n--- 用 MemoryPool 的 with 块 ---")
with cp.cuda.using_allocator(cp.cuda.MemoryPool().malloc):
    tmp = cp.empty(500 * 1024**2 // 4, dtype=cp.float32)
    show_mem("with 块内")
show_mem("with 块外（池子已销毁，显存归还）")


def mem_report(tag=""):
    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.Device(0).mem_info
    print(
        f"[{tag}] 真实占用 {pool.used_bytes() / 1024**2:.0f}MB, "
        f"池缓存 {pool.total_bytes() / 1024**2:.0f}MB, "
        f"设备剩余 {free / 1024**2:.0f}MB"
    )


# 在关键节点调用
mem_report("处理前")
# ... 处理 ...
mem_report("处理后")
