"""后台推理线程。放在独立模块里，主窗口和对比窗口共用，避免循环导入。"""

from __future__ import annotations

import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from src.grade import grade


class InferenceWorker(QThread):
    """跑一次推理 + 判级，结果通过信号回主线程。"""

    finished = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, detector, image, conf: float) -> None:
        super().__init__()
        self.detector = detector
        self.image = image
        self.conf = conf

    def run(self) -> None:  # noqa: D102
        try:
            result = self.detector.predict(self.image, conf=self.conf)
            verdict = grade(result.detections, result.image_size)
            self.finished.emit(result, verdict)
        except Exception:  # noqa: BLE001 - 界面要显示错误而不是崩掉
            self.failed.emit(traceback.format_exc())
