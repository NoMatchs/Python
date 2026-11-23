# 需要以管理员权限运行
from HardView.LiveView import PyTempCpu

try:
    cpu_temp = PyTempCpu()
    print(f"CPU 温度: {cpu_temp.get_temp():.1f}°C")
except Exception as e:
    print(f"读取失败: {e}")
