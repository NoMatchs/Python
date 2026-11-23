import psutil
import time
import csv
from datetime import datetime

with open("cpu_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "cpu_percent"])

    for _ in range(60):  # 采集 60 次
        cpu = psutil.cpu_percent(interval=1)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        writer.writerow([now, cpu])
        print(f"{now}  CPU: {cpu}%")
