import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

SNAP_DISTANCE = 20  # 距离边缘多少像素内自动吸附


class GpuWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(30,30,30,210); border-radius: 8px;")
        self.resize(260, 210)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        def make_label(text, color, size=12):
            lb = QLabel(text)
            lb.setStyleSheet(f"color: {color}; font: bold {size}px Consolas;")
            layout.addWidget(lb)
            return lb

        title = QLabel("RTX 4060 Laptop")
        title.setStyleSheet("color: #888; font: bold 10px Consolas;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.temp_label = make_label("温度: --", "#00FF00", 13)
        self.power_label = make_label("功耗: --", "#00BFFF", 13)
        self.mem_label = make_label("显存: --", "#FFD700", 13)
        self.util_label = make_label("利用率: --", "#FF69B4", 13)
        self.clock_label = make_label("频率: --", "#00FFAA", 13)
        self.thresh_label = make_label("温度墙: --", "#FFA500", 13)
        self.pcie_label = make_label("PCIe: --", "#BBBBBB", 13)

        self._drag_pos = None

    # ---------- 拖拽 + 吸附 ----------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.snap_to_edge()
            self._drag_pos = None
            event.accept()

    def snap_to_edge(self):
        screen = QApplication.primaryScreen().availableGeometry()
        x, y = self.x(), self.y()
        w, h = self.width(), self.height()

        if x < screen.left() + SNAP_DISTANCE:
            x = screen.left()
        if x + w > screen.right() - SNAP_DISTANCE:
            x = screen.right() - w
        if y < screen.top() + SNAP_DISTANCE:
            y = screen.top()
        if y + h > screen.bottom() - SNAP_DISTANCE:
            y = screen.bottom() - h

        self.move(x, y)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    # ---------- 更新数据 ----------
    def update_data(self):
        temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
        power = nvmlDeviceGetPowerUsage(handle) / 1000

        mem = nvmlDeviceGetMemoryInfo(handle)
        mem_used = mem.used / 1024**2
        mem_total = mem.total / 1024**2
        mem_pct = mem.used / mem.total * 100

        util = nvmlDeviceGetUtilizationRates(handle)

        try:
            core_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_GRAPHICS)
            mem_clock = nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)
            self.clock_label.setText(
                f"频率: 核心 {core_clock} MHz / 显存 {mem_clock} MHz"
            )
        except NVMLError:
            self.clock_label.setText("频率: 不支持读取")

        try:
            slow = nvmlDeviceGetTemperatureThreshold(
                handle, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN
            )
            shut = nvmlDeviceGetTemperatureThreshold(
                handle, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN
            )
            self.thresh_label.setText(f"温度墙: 降频 {slow}°C / 关机 {shut}°C")
        except NVMLError:
            self.thresh_label.setText("温度墙: 不支持读取")

        try:
            width = nvmlDeviceGetCurrPcieLinkWidth(handle)
            gen = nvmlDeviceGetCurrPcieLinkGeneration(handle)
            self.pcie_label.setText(f"PCIe: Gen{gen} x{width}")
        except NVMLError:
            self.pcie_label.setText("PCIe: 不支持读取")

        self.temp_label.setText(f"温度: {temp}°C")
        self.power_label.setText(f"功耗: {power:.1f} W")
        self.mem_label.setText(
            f"显存: {mem_used:.0f}/{mem_total:.0f} MB ({mem_pct:.0f}%)"
        )
        self.util_label.setText(f"利用率: GPU {util.gpu}%  显存 {util.memory}%")

        if temp >= 85:
            self.temp_label.setStyleSheet("color: red; font: bold 13px Consolas;")
        elif temp >= 80:
            self.temp_label.setStyleSheet("color: orange; font: bold 13px Consolas;")
        else:
            self.temp_label.setStyleSheet("color: #00FF00; font: bold 13px Consolas;")


app = QApplication(sys.argv)
win = GpuWidget()

timer = QTimer()
timer.timeout.connect(win.update_data)
timer.start(1000)
win.update_data()

win.show()
sys.exit(app.exec_())
