"""双模型对比视图：同一张图，左边一个模型、右边另一个，直接看判级差异。

对应论文里的对比实验：YOLO11n vs 自研 ResNet50+FPN+CBAM。
同一张图、同一个置信度阈值，两个模型各自出框、各自判级，
底部给出"差了哪几级、差在哪项判据"的总结。

    python -m src.ui.compare
    python -m src.ui.compare --image data/msdd/images/val/xxx.jpg

主窗口里点"模型对比"也会打开这个窗口。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.config import repo_path
from src.detectors import load_detector
from src.grade import GradeResult
from src.ui.theme import CLASS_COLORS, badge_style
from src.ui.worker import InferenceWorker

DEFAULT_A = "runs/baseline/weights/best.pt"
DEFAULT_B = "runs/cnn-cbam/best.pt"


def friendly_name(weights: str) -> str:
    """把权重路径翻译成人看的名字——两边都叫 best.pt，不区分根本分不清谁是谁。

    注意权重可能放在 runs/<name>/weights/best.pt，也可能直接是 runs/<name>/best.pt，
    所以按整条路径判断，而不是只看父目录名。
    """
    path = Path(weights).as_posix().lower()
    if "cnn" in path:
        return "自研 CNN（ResNet50+FPN+CBAM）"
    if "seg" in path:
        return "YOLO11n-seg（分割）"
    if "neu" in path:
        return "YOLO11n（NEU-DET）"
    return "YOLO11n（检测）"


class ModelPane(QWidget):
    """一个模型的展示面板：标题 + 标注图 + 等级 + 处置 + 关键指标。"""

    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            "font-size:13px; font-weight:bold; color:#f0f0f0; background:#2a2a2a;"
            "border-radius:4px; padding:6px;"
        )
        layout.addWidget(self.title_label)

        self.canvas = QLabel("等待图片")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setMinimumSize(280, 220)
        self.canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.canvas.setStyleSheet("background:#111; color:#888;")
        layout.addWidget(self.canvas, 1)

        self.badge = QLabel("—")
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setStyleSheet(badge_style(-1))
        layout.addWidget(self.badge)

        self.action = QLabel("处置：—")
        self.action.setAlignment(Qt.AlignCenter)
        self.action.setWordWrap(True)
        self.action.setStyleSheet(
            "background:#1e1e1e; color:#e5e5e5; border-radius:6px; padding:6px; font-size:12px;"
        )
        layout.addWidget(self.action)

        self.stats = QLabel("")
        self.stats.setAlignment(Qt.AlignCenter)
        self.stats.setStyleSheet("color:#999; font-size:11px;")
        layout.addWidget(self.stats)

        self.current_pixmap = None
        self.verdict: GradeResult | None = None

    def set_pixmap(self, pixmap) -> None:
        self.current_pixmap = pixmap
        self._scale()

    def _scale(self) -> None:
        if self.current_pixmap is None:
            return
        self.canvas.setPixmap(
            self.current_pixmap.scaled(self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._scale()

    def show_result(self, result, verdict: GradeResult) -> None:
        self.verdict = verdict
        self.badge.setText(verdict.level_name)
        self.badge.setStyleSheet(badge_style(verdict.level))
        self.action.setText(f"处置：{verdict.action}")
        self.stats.setText(
            f"缺陷 {int(verdict.features.get('count', 0))} 处 · "
            f"覆盖 {verdict.features.get('area_ratio', 0) * 100:.2f}% · "
            f"{result.latency_ms:.0f} ms"
        )


class CompareWindow(QMainWindow):
    def __init__(self, weights_a: str, weights_b: str, conf: float = 0.25) -> None:
        super().__init__()
        self.setWindowTitle("模型对比：同一张图，两个模型各自判级")
        self.resize(1360, 860)
        self.image: np.ndarray | None = None
        self.workers: list[InferenceWorker] = []

        self.detector_a = load_detector(weights_a)
        self.detector_b = load_detector(weights_b)

        central = QWidget()
        layout = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        self.open_button = QPushButton("打开图片")
        self.open_button.clicked.connect(self.open_image)
        self.rerun_button = QPushButton("重新检测")
        self.rerun_button.clicked.connect(self.run_both)
        toolbar.addWidget(self.open_button)
        toolbar.addWidget(self.rerun_button)
        toolbar.addStretch(1)
        toolbar.addWidget(QLabel("置信度阈值"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(int(conf * 100))
        self.conf_slider.setFixedWidth(150)
        self.conf_value = QLabel(f"{conf:.2f}")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_value.setText(f"{v / 100:.2f}"))
        toolbar.addWidget(self.conf_slider)
        toolbar.addWidget(self.conf_value)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        self.pane_a = ModelPane(f"A · {friendly_name(weights_a)}")
        self.pane_b = ModelPane(f"B · {friendly_name(weights_b)}")
        splitter.addWidget(self.pane_a)
        splitter.addWidget(self.pane_b)
        splitter.setSizes([680, 680])
        layout.addWidget(splitter, 1)

        self.summary = QLabel("打开一张图片后，这里会给出两个模型的判级差异")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(
            "background:#1e1e1e; color:#eee; border-radius:6px; padding:10px; font-size:13px;"
        )
        layout.addWidget(self.summary)

        self.setCentralWidget(central)
        self.statusBar().showMessage(f"A={weights_a}   B={weights_b}")

    # ------------------------------------------------------------- actions
    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(repo_path("data")), "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        from PIL import Image

        self.image = np.asarray(Image.open(path).convert("RGB"))
        self.setWindowTitle(f"模型对比 · {path.name}")
        plain = self._to_pixmap(self.image)
        for pane in (self.pane_a, self.pane_b):
            pane.set_pixmap(plain)
        self.run_both()

    def _to_pixmap(self, array) -> QPixmap:
        height, width, _ = array.shape
        image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def run_both(self) -> None:
        if self.image is None:
            return
        conf = self.conf_slider.value() / 100
        self.statusBar().showMessage("两个模型同时推理中…")
        self.workers = [
            InferenceWorker(self.detector_a, self.image, conf),
            InferenceWorker(self.detector_b, self.image, conf),
        ]
        self.workers[0].finished.connect(lambda r, v: self.on_finished("a", r, v))
        self.workers[1].finished.connect(lambda r, v: self.on_finished("b", r, v))
        for worker in self.workers:
            worker.failed.connect(lambda message: QMessageBox.critical(self, "推理失败", message))
            worker.start()

    def on_finished(self, key: str, result, verdict: GradeResult) -> None:
        pane = self.pane_a if key == "a" else self.pane_b
        self._draw(pane, result)
        pane.show_result(result, verdict)
        if self.pane_a.verdict is not None and self.pane_b.verdict is not None:
            self.update_summary()

    def _draw(self, pane: ModelPane, result) -> None:
        if self.image is None:
            return
        height, width, _ = self.image.shape
        image = QImage(self.image.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        font = QFont()
        font.setPointSize(max(8, width // 70))
        painter.setFont(font)
        for detection in result.detections:
            color = QColor(CLASS_COLORS[detection.cls_id % len(CLASS_COLORS)])
            pen = QPen(color)
            pen.setWidth(max(2, width // 260))
            painter.setPen(pen)
            x0, y0, x1, y1 = detection.box
            painter.drawRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            painter.setPen(QPen(color))
            painter.drawText(int(x0) + 3, int(max(10, y0 - 4)), f"{detection.cls_name} {detection.confidence:.2f}")
        painter.end()
        pane.set_pixmap(pixmap)

    def update_summary(self) -> None:
        a = self.pane_a.verdict
        b = self.pane_b.verdict
        if a is None or b is None:
            return
        delta = b.level - a.level
        if delta == 0:
            verdict_text = "两个模型判级一致"
        else:
            stricter = "B（自研 CNN）" if delta > 0 else "A（YOLO）"
            verdict_text = f"{stricter} 判得更严，相差 {abs(delta)} 级"
        area_a = a.features.get("area_ratio", 0) * 100
        area_b = b.features.get("area_ratio", 0) * 100
        self.summary.setText(
            f"{verdict_text}　|　"
            f"A：{a.level_name}（{int(a.features.get('count', 0))} 处，覆盖 {area_a:.2f}%）　 vs　 "
            f"B：{b.level_name}（{int(b.features.get('count', 0))} 处，覆盖 {area_b:.2f}%）"
        )
        self.statusBar().showMessage("完成")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ui.compare", description="双模型对比")
    parser.add_argument("--weights-a", default=DEFAULT_A)
    parser.add_argument("--weights-b", default=DEFAULT_B)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--image")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    window = CompareWindow(args.weights_a, args.weights_b, args.conf)
    window.show()
    if args.image:
        path = Path(args.image)
        if path.exists():
            window.load_image(path)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
