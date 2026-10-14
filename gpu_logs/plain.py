import pandas as pd
import matplotlib.pyplot as plt
import os

# 用脚本所在目录来定位 CSV，避免路径歧义
base = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base, "gpu_20261013.csv")

print(f"读取: {csv_path}")
df = pd.read_csv(csv_path)

print(df.head())
print(f"共 {len(df)} 条记录")

df["温度(°C)"].plot()
plt.title("GPU 温度曲线")
plt.xlabel("采样点")
plt.ylabel("温度 (°C)")
plt.grid(True)
plt.show()  # 弹窗显示
# plt.savefig("temp_curve.png")  # 如果想保存成图片，用这行替代上面那行
