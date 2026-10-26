#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reels Interactive Diagnostic Video Player
-----------------------------------------
功能：
1. 視窗上方播放 Reels 影片 (.mp4)。
2. 視窗下方同步顯示三大秒級診斷圖表 (微表情/B-roll、音訊/Hook、面部光影)。
3. 一條顯眼的動態紅色時間標尺 (Cursor) 隨播放進度即時前進。
4. 支援「點擊圖表任意秒數，影片立即跳轉播放」。
"""

import sys
import numpy as np
from pathlib import Path

# GUI 與多媒體函式庫 (優先支援 PyQt5，向下相容)
from PyQt5.QtCore import QUrl, QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QSplitter
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

# 嵌入式 Matplotlib 支援
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class InteractiveReelsPlayer(QMainWindow):
    def __init__(self, video_path: str, diag_data: dict, benchmarks: dict):
        super().__init__()
        self.video_path = Path(video_path)
        self.diag = diag_data
        self.benchmarks = benchmarks
        self.total_duration_sec = self.diag.get("duration", 60)

        self.setWindowTitle(f"Reels 秒級 AI 體檢播放器 - {self.video_path.name}")
        self.resize(1100, 920)

        self.init_ui()
        self.init_player()
        self.plot_graphs()

    def init_ui(self):
        # 主佈局採用垂直分割視窗 (上部影片，下部圖表)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # -----------------------------
        # 上半部：影片播放畫面與播放控制列
        # -----------------------------
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(320)
        top_layout.addWidget(self.video_widget)

        # 控制列 (播放/暫停按鈕、進度條、當前時間顯示)
        control_bar = QHBoxLayout()
        self.play_btn = QPushButton("▶ 播放 (Space)")
        self.play_btn.setFixedWidth(120)
        self.play_btn.clicked.connect(self.toggle_play)
        control_bar.addWidget(self.play_btn)

        self.time_label = QLabel("00:00 / 00:00")
        control_bar.addWidget(self.time_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(self.total_duration_sec * 1000))
        self.slider.sliderMoved.connect(self.set_position)
        control_bar.addWidget(self.slider)

        top_layout.addLayout(control_bar)
        splitter.addWidget(top_container)

        # -----------------------------
        # 下半部：嵌入式 Matplotlib 三大診斷圖表
        # -----------------------------
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(10, 5.2), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect("button_press_event", self.on_graph_clicked)
        bottom_layout.addWidget(self.canvas)

        splitter.addWidget(bottom_container)
        splitter.setStretchFactor(0, 4)  # 影片佔 4 成比例
        splitter.setStretchFactor(1, 6)  # 圖表佔 6 成比例

    def init_player(self):
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(str(self.video_path))))

        # 定時器以 50ms 頻率更新時間軸與紅線遊標
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.sync_cursor_with_video)
        self.timer.start()

        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)

    # -----------------------------------------------------------------------
    # 繪製三大診斷圖表 (並標記垂直紅色進度線)
    # -----------------------------------------------------------------------
    def plot_graphs(self):
        self.fig.clear()
        T = self.total_duration_sec
        secs = np.arange(T)
        raw_np = self.diag["raw_np"][:T]

        self.axes = self.fig.subplots(3, 1, sharex=True)
        self.fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.35)

        # ----------------- 子圖 1: 微表情與 B-roll 建議 -----------------
        ax1 = self.axes[0]
        ax1.plot(secs, raw_np[:, 4], color="#1f77b4", lw=2, label="嘴角/眼周微表情 (Key Mu)")
        ax1.axhline(self.benchmarks["key_mu_p30"], color="red", ls="--", alpha=0.7, label="最低表情活躍線")
        for idx, (st, ed) in enumerate(self.diag.get("broll_intervals", [])):
            ax1.axvspan(st, ed, color="orange", alpha=0.35, label="建議加 B-roll" if idx == 0 else None)
        ax1.set_ylabel("微表情強度", fontweight="bold", fontsize=9)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, ls=":", alpha=0.5)
        ax1.set_title("【微表情張力與 B-roll 切入點】(點擊任意秒數跳轉)", fontsize=10, fontweight="bold", loc="left")

        # ----------------- 子圖 2: 語音能量與 Transformer 注意力 -----------------
        ax2 = self.axes[1]
        ax2.bar(secs, raw_np[:, 6], color="#2ca02c", alpha=0.4, width=0.6, label="語音重音 (Audio Peak)")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(secs, self.diag["temporal_importance"][:T], color="#9467bd", lw=2, label="AI 注意力權重")
        ax2.axvspan(0, min(3, T), color="gold", alpha=0.25, label="首 3 秒 Hook 區")
        ax2.set_ylabel("聲能脈衝", fontweight="bold", color="#2ca02c", fontsize=9)
        ax2_twin.set_ylabel("注意力", fontweight="bold", color="#9467bd", fontsize=9)
        ax2.legend(loc="upper left", fontsize=8)
        ax2_twin.legend(loc="upper right", fontsize=8)
        ax2.grid(True, ls=":", alpha=0.5)

        # ----------------- 子圖 3: 面部打光柔和度 -----------------
        ax3 = self.axes[2]
        ax3.plot(secs, raw_np[:, 1], color="#d62728", lw=1.8, label="面部 RMS 對比 (越低越柔和)")
        ax3.axhline(self.benchmarks["face_rms_p70"], color="#d62728", ls="--", alpha=0.7, label="硬光陰影警戒線")
        ax3.set_ylabel("RMS 對比", fontweight="bold", color="#d62728", fontsize=9)
        ax3.set_xlabel("影片秒數 (Seconds)", fontweight="bold", fontsize=9)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, ls=":", alpha=0.5)
        ax3.set_xlim(0, T)

        # 初始化三個子圖中的垂直紅色標尺線 (Cursor)
        self.cursor_lines = []
        for ax in self.axes:
            line = ax.axvline(x=0, color="#ff0000", lw=2.2, linestyle="-", zorder=10)
            self.cursor_lines.append(line)

        self.canvas.draw()

    # -----------------------------------------------------------------------
    # 播放與即時連動邏輯
    # -----------------------------------------------------------------------
    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.play_btn.setText("▶ 播放 (Space)")
        else:
            self.player.play()
            self.play_btn.setText("⏸ 暫停 (Space)")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_play()

    def set_position(self, position):
        self.player.setPosition(position)

    def on_position_changed(self, position):
        if not self.slider.isSliderDown():
            self.slider.setValue(position)

    def on_duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def sync_cursor_with_video(self):
        """核心：每 50 毫秒將紅線移動至影片當前的真實秒數"""
        cur_pos_ms = self.player.position()
        cur_sec = cur_pos_ms / 1000.0

        dur_ms = self.player.duration() or (self.total_duration_sec * 1000)
        cur_str = f"{int(cur_sec//60):02d}:{int(cur_sec%60):02d}"
        tot_str = f"{int((dur_ms/1000)//60):02d}:{int((dur_ms/1000)%60):02d}"
        self.time_label.setText(f"{cur_str} / {tot_str}")

        # 更新紅線 X 座標
        if hasattr(self, "cursor_lines"):
            for line in self.cursor_lines:
                line.set_xdata([cur_sec, cur_sec])
            self.canvas.draw_idle()

    def on_graph_clicked(self, event):
        """使用者點擊圖表任意位置時，自動跳轉影片到該秒數"""
        if event.xdata is not None and event.inaxes in self.axes:
            target_sec = max(0.0, min(float(event.xdata), float(self.total_duration_sec)))
            self.player.setPosition(int(target_sec * 1000))
            if self.player.state() != QMediaPlayer.PlayingState:
                self.player.play()
                self.play_btn.setText("⏸ 暫停 (Space)")


def launch_inspector(video_path: str, diag_data: dict, benchmarks: dict):
    """彈出審片播放器視窗入口"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    player = InteractiveReelsPlayer(video_path, diag_data, benchmarks)
    player.show()
    app.exec_()