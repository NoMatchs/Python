from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

# 基础标识
name = nvmlDeviceGetName(handle)
if isinstance(name, bytes):
    name = name.decode()
print(f"设备: {name}")
print(f"驱动版本: {nvmlSystemGetDriverVersion()}")

# 显存
mem = nvmlDeviceGetMemoryInfo(handle)
print(f"显存: {mem.used / 1024**2:.0f} MB / {mem.total / 1024**2:.0f} MB")

# 利用率（核心 + 显存带宽）
util = nvmlDeviceGetUtilizationRates(handle)
print(f"利用率: GPU {util.gpu}% / 显存带宽 {util.memory}%")

# 温度
temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
print(f"温度: {temp}°C")

# 功耗
power = nvmlDeviceGetPowerUsage(handle) / 1000
print(f"功耗: {power:.1f} W")

# 时钟频率（核心 / 显存）
core_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
mem_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)
print(f"频率: 核心 {core_clock} MHz / 显存 {mem_clock} MHz")

# PCIe 链路
gen = nvmlDeviceGetCurrPcieLinkGeneration(handle)
width = nvmlDeviceGetCurrPcieLinkWidth(handle)
print(f"PCIe: Gen{gen} x{width}")

# 温度阈值（降频 / 关机）
slow = nvmlDeviceGetTemperatureThreshold(handle, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
shut = nvmlDeviceGetTemperatureThreshold(handle, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
print(f"温度墙: 降频 {slow}°C / 关机 {shut}°C")

nvmlShutdown()
