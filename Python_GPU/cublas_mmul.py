import os

# 关键：告诉 Python 去这个目录下寻找 cublas 及其依赖的 DLL
os.add_dll_directory(r"D:\conda_envs\i5xia_CUDA\Lib\site-packages\nvidia\cublas\bin")

import ctypes
import cupy as cp
from cupyx.profiler import benchmark


def main():
    # 现在可以直接用文件名加载了
    libcublas = ctypes.CDLL("cublas64_12.dll")

    N = 2048
    a_gpu = cp.random.rand(N, N, dtype=cp.float32)
    b_gpu = cp.random.rand(N, N, dtype=cp.float32)
    c_gpu = cp.zeros((N, N), dtype=cp.float32)

    libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    libcublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]

    # 修正为准确的 14 个参数类型
    libcublas.cublasSgemm_v2.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,  # transa
        ctypes.c_int,  # transb
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.c_int,  # k
        ctypes.c_void_p,  # alpha
        ctypes.c_void_p,  # A
        ctypes.c_int,  # lda
        ctypes.c_void_p,  # B
        ctypes.c_int,  # ldb
        ctypes.c_void_p,  # beta
        ctypes.c_void_p,  # C
        ctypes.c_int,  # ldc
    ]

    handle = ctypes.c_void_p()
    libcublas.cublasCreate_v2(ctypes.byref(handle))

    CUBLAS_OP_N = 0
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    def run_cublas_py(a, b, c):
        libcublas.cublasSgemm_v2(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            N,
            N,
            ctypes.byref(alpha),
            b.data.ptr,
            N,  # A 位置传 b（行主序转置技巧）
            a.data.ptr,
            N,  # B 位置传 a
            ctypes.byref(beta),
            c.data.ptr,
            N,
        )

    # 预热 + 验证
    run_cublas_py(a_gpu, b_gpu, c_gpu)
    cp.cuda.Stream.null.synchronize()

    expected = cp.dot(a_gpu, b_gpu)
    print(f"结果误差: {cp.max(cp.abs(c_gpu - expected)):.5f}")

    # 测速
    res = benchmark(run_cublas_py, (a_gpu, b_gpu, c_gpu), n_repeat=10, n_warmup=2)
    print(f"cuBLAS C++ 调用耗时:\n{res}")

    libcublas.cublasDestroy_v2(handle)


if __name__ == "__main__":
    main()
