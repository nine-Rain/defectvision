"""PyQt5 桌面端：导入图片 → 缺陷检测 → 等级判定 → 报告导出。

对应论文的"系统应用层"。启动：

    python -m src.ui
    python -m src.ui --weights runs/baseline-seg/weights/best.pt
    python -m src.ui --weights runs/cnn-cbam/best.pt      # 我们自己的 ResNet-FPN-CBAM

设计上把推理放在 QThread 里，界面不会卡；判级规则走 configs/grading.yaml，
工艺人员改阈值不需要重训模型。
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.config import repo_path
from src.detectors import DetectionResult, load_detector
from src.grade import GradeResult, format_report, grade

CLASS_COLORS = [
    "#ffd60a", "#ff3b30", "#30d158", "#0a84ff", "#ff5c8d", "#bf5af2",
    "#64d2ff", "#ff9f0a", "#5e5ce6", "#ac8e68",
]
LEVEL_STYLE = {
    0: ("#30d158", "合格"),
    1: ("#30d158", "I级 轻微"),
    2: ("#ffd60a", "II级 中度"),
    3: ("#ff9f0a", "III级 严重"),
    4: ("#ff3b30", "IV级 致命"),
}


class InferenceWorker(QThread):
    """在后台线程里跑推理，避免界面假死。"""

    finished = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, detector, image, conf: float) -> None:
        super().__init__()
        self.detector = detector
        self.image = image
        self.conf = conf

    def run(self) -> None:  # noqa: D102
        try:
            result: DetectionResult = self.detector.predict(self.image, conf=self.conf)
            verdict = grade(result.detections, result.image_size)
            self.finished.emit(result, verdict)
        except Exception:  # noqa: BLE001 - 界面要显示错误而不是崩掉
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self, weights: str, conf: float = 0.25) -> None:
        super().__init__()
        self.setWindowTitle("表面缺陷检测与等级判定系统")
        self.resize(1280, 800)

        self.weights = weights
        self.detector = load_detector(weights)
        self.names = self.detector.names
        self.image = None  # numpy RGB
        self.image_path: Path | None = None
        self.current_pixmap: QPixmap | None = None  # 已经画好框/掩码的图
        self.last_result: DetectionResult | None = None
        self.last_verdict: GradeResult | None = None
        self.worker: InferenceWorker | None = None

        self._build_ui(conf)

    # ------------------------------------------------------------------ UI
    def _build_ui(self, conf: float) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        self.open_button = QPushButton("打开图片")
        self.open_button.clicked.connect(self.open_image)
        self.batch_button = QPushButton("批量检测…")
        self.batch_button.clicked.connect(self.batch_process)
        self.detect_button = QPushButton("重新检测")
        self.detect_button.clicked.connect(self.run_inference)
        self.export_button = QPushButton("导出报告")
        self.export_button.clicked.connect(self.export_report)
        self.export_button.setEnabled(False)

        toolbar.addWidget(self.open_button)
        toolbar.addWidget(self.batch_button)
        toolbar.addWidget(self.detect_button)
        toolbar.addStretch(1)
        toolbar.addWidget(QLabel("置信度阈值"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(int(conf * 100))
        self.conf_slider.setFixedWidth(160)
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v / 100:.2f}"))
        self.conf_label = QLabel(f"{conf:.2f}")
        toolbar.addWidget(self.conf_slider)
        toolbar.addWidget(self.conf_label)
        toolbar.addWidget(self.export_button)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        self.canvas = QLabel("请打开一张待检图片")
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setMinimumSize(320, 240)
        # 忽略 QLabel 根据 pixmap 推导出的 sizeHint，否则窗口会被图片尺寸反推而挤掉工具栏
        self.canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.canvas.setStyleSheet("background:#111; color:#888; font-size:16px;")
        splitter.addWidget(self.canvas)
        splitter.addWidget(self._build_side_panel())
        splitter.setSizes([880, 380])
        layout.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage(f"模型: {Path(self.weights).name}  类别: {len(self.names)}")

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.level_label = QLabel("—")
        self.level_label.setAlignment(Qt.AlignCenter)
        self.level_label.setStyleSheet(
            "background:#333; color:#fff; border-radius:8px; padding:18px; font-size:24px; font-weight:bold;"
        )
        layout.addWidget(self.level_label)

        self.score_label = QLabel("综合得分 —")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("font-size:14px; color:#aaa;")
        layout.addWidget(self.score_label)

        group = QGroupBox("缺陷清单")
        group_layout = QVBoxLayout(group)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["类别", "置信度", "面积占比", "严重度"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(160)
        group_layout.addWidget(self.table)
        layout.addWidget(group, 1)

        self.reason_label = QLabel("")
        self.reason_label.setWordWrap(True)
        self.reason_label.setMinimumHeight(96)
        self.reason_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.reason_label.setStyleSheet("color:#aaa; font-size:12px;")
        layout.addWidget(self.reason_label)
        return panel

    # -------------------------------------------------------------- actions
    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(repo_path("data")), "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        self.image = np.asarray(image)
        self.image_path = path
        self.show_pixmap(self.image)
        self.run_inference()

    def run_inference(self) -> None:
        if self.image is None:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        conf = self.conf_slider.value() / 100
        self.statusBar().showMessage("检测中…")
        self.worker = InferenceWorker(self.detector, self.image, conf)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_finished(self, result: DetectionResult, verdict: GradeResult) -> None:
        self.last_result = result
        self.last_verdict = verdict
        self.draw_overlay(result)
        self.update_panel(result, verdict)
        self.export_button.setEnabled(True)
        self.statusBar().showMessage(
            f"完成：{len(result.detections)} 个目标，{result.latency_ms:.1f} ms"
        )

    def on_failed(self, message: str) -> None:
        QMessageBox.critical(self, "推理失败", message)
        self.statusBar().showMessage("推理失败")

    # ------------------------------------------------------------- rendering
    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().resizeEvent(event)
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        if self.current_pixmap is None or self.canvas is None:
            return
        self.canvas.setPixmap(
            self.current_pixmap.scaled(
                self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def show_pixmap(self, array) -> None:
        height, width, _ = array.shape
        image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self.current_pixmap = QPixmap.fromImage(image)
        self._apply_pixmap()

    def draw_overlay(self, result: DetectionResult) -> None:
        if self.image is None:
            return
        height, width, _ = self.image.shape
        image = QImage(self.image.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, False)
        font = QFont()
        font.setPointSize(max(9, width // 60))
        painter.setFont(font)

        for detection in result.detections:
            color = QColor(CLASS_COLORS[detection.cls_id % len(CLASS_COLORS)])
            x0, y0, x1, y1 = detection.box
            pen = QPen(color)
            pen.setWidth(max(2, width // 250))
            painter.setPen(pen)
            painter.drawRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))

            label = f"{detection.cls_name} {detection.confidence:.2f}"
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(label) + 8
            text_height = metrics.height()
            painter.fillRect(int(x0), int(max(0, y0 - text_height)), text_width, text_height, color)
            painter.setPen(QPen(QColor("#111")))
            painter.drawText(int(x0) + 4, int(max(text_height - 3, y0 - 4)), label)
        painter.end()

        self.current_pixmap = pixmap
        self._apply_pixmap()

    def update_panel(self, result: DetectionResult, verdict: GradeResult) -> None:
        color, text = LEVEL_STYLE.get(verdict.level, ("#888", verdict.level_name))
        self.level_label.setText(text)
        self.level_label.setStyleSheet(
            f"background:{color}; color:#111; border-radius:8px; padding:18px;"
            "font-size:24px; font-weight:bold;"
        )
        self.score_label.setText(f"综合得分 {verdict.score:.1f} / 100")
        self.reason_label.setText("\n".join(f"· {r}" for r in verdict.reasons))

        self.table.setRowCount(len(verdict.defects))
        for row, defect in enumerate(verdict.defects):
            values = [
                defect["class"],
                f"{defect['confidence']:.3f}",
                f"{defect['area_ratio'] * 100:.2f}%",
                str(defect["severity"]),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, column, item)

    # --------------------------------------------------------------- export
    def export_report(self) -> None:
        if self.last_verdict is None:
            return
        default = str(repo_path("runs") / "reports" / "report.txt")
        path, _ = QFileDialog.getSaveFileName(self, "导出报告", default, "文本 (*.txt)")
        if not path:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        report = format_report(
            self.last_verdict,
            image_name=self.image_path.name if self.image_path else "",
            model_name=Path(self.weights).name,
            elapsed_ms=self.last_result.latency_ms if self.last_result else None,
        )
        target.write_text(report, encoding="utf-8")
        if self.current_pixmap is not None:
            self.current_pixmap.save(str(target.with_suffix(".jpg")), "JPG", 92)
        QMessageBox.information(self, "已导出", f"报告已保存：\n{target}")

    def batch_process(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not directory:
            return
        images = sorted(
            p for p in Path(directory).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not images:
            QMessageBox.warning(self, "没有图片", "该目录下没有可识别的图片")
            return

        out_dir = Path(directory) / "reports"
        out_dir.mkdir(exist_ok=True)
        rows = []
        conf = self.conf_slider.value() / 100
        from PIL import Image

        for index, path in enumerate(images, start=1):
            self.statusBar().showMessage(f"批量检测 {index}/{len(images)}: {path.name}")
            QApplication.processEvents()
            image = np.asarray(Image.open(path).convert("RGB"))
            result = self.detector.predict(image, conf=conf)
            verdict = grade(result.detections, result.image_size)
            (out_dir / f"{path.stem}.txt").write_text(
                format_report(verdict, path.name, Path(self.weights).name, result.latency_ms),
                encoding="utf-8",
            )
            rows.append(
                {
                    "image": path.name,
                    "defects": len(result.detections),
                    "score": round(verdict.score, 2),
                    "level": verdict.level_name,
                }
            )

        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "defects", "score", "level"])
            writer.writeheader()
            writer.writerows(rows)
        self.statusBar().showMessage(f"批量完成：{len(images)} 张，报告在 {out_dir}")
        QMessageBox.information(self, "批量完成", f"共 {len(images)} 张，报告目录：\n{out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ui", description="桌面端缺陷检测系统")
    parser.add_argument("--weights", default="runs/baseline/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    window = MainWindow(args.weights, args.conf)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
