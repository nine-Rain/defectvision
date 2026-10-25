"""检测器的选择与推理封装。

桌面端（PyQt）和脚本都要"给一张图，拿回一串检测结果"，但底层可能是
YOLO 的 .pt/.onnx，也可能是我们自己训练的 ResNet-FPN-CBAM。这里抹平差异。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    confidence: float
    box: tuple[float, float, float, float]  # xyxy 像素
    mask: np.ndarray | None = None

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    latency_ms: float = 0.0


class BaseDetector:
    names: dict[int, str]

    def predict(self, image: np.ndarray, conf: float = 0.25) -> DetectionResult:
        raise NotImplementedError


class YoloDetector(BaseDetector):
    def __init__(self, weights: Path) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        raw = self.model.names
        self.names = {int(k): str(v) for k, v in raw.items()}

    def predict(self, image: np.ndarray, conf: float = 0.25) -> DetectionResult:
        import time

        start = time.perf_counter()
        result = self.model.predict(image, conf=conf, verbose=False)[0]
        latency = (time.perf_counter() - start) * 1000

        detections: list[Detection] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None
            for index in range(len(boxes)):
                detections.append(
                    Detection(
                        cls_id=int(classes[index]),
                        cls_name=self.names.get(int(classes[index]), str(classes[index])),
                        confidence=float(scores[index]),
                        box=tuple(float(v) for v in boxes[index]),
                        mask=masks[index] if masks is not None else None,
                    )
                )
        return DetectionResult(detections, (result.orig_shape[1], result.orig_shape[0]), latency)


class TorchvisionDetector(BaseDetector):
    """我们自己的 ResNet-FPN-CBAM 检测器。"""

    def __init__(self, weights: Path, names: dict[int, str], device: str = "auto") -> None:
        import torch

        from src.models.detector import build_detector

        self.device = self._resolve_device(device)
        checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
        meta = checkpoint.get("meta", {})
        self.names = meta.get("names", names)

        self.model = build_detector(
            num_classes=meta.get("num_classes", len(self.names) + 1),
            backbone=meta.get("backbone", "resnet50"),
            use_cbam=meta.get("use_cbam", True),
            min_size=meta.get("imgsz", 512),
            max_size=meta.get("imgsz", 512),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        self.imgsz = meta.get("imgsz", 512)

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if device != "auto":
            # 命令行习惯写 "0"，但 torch 要 "cuda" 或 "cuda:0"
            return f"cuda:{device}" if device.isdigit() else device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, image: np.ndarray, conf: float = 0.25) -> DetectionResult:
        import time

        import torch

        start = time.perf_counter()
        # 先转 float 再交给 torch：uint8 -> float32 会产生新数组，避免 torch 对只读数组报警告
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        with torch.inference_mode():
            output = self.model([tensor.to(self.device)])[0]
        latency = (time.perf_counter() - start) * 1000

        detections: list[Detection] = []
        for box, score, label in zip(
            output["boxes"], output["scores"], output["labels"], strict=True
        ):
            if float(score) < conf:
                continue
            cls_id = int(label) - 1  # 训练时 +1 过，这里还原
            detections.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=self.names.get(cls_id, str(cls_id)),
                    confidence=float(score),
                    box=tuple(float(v) for v in box),
                )
            )
        height, width = image.shape[:2]
        return DetectionResult(detections, (width, height), latency)


def load_detector(weights: str | Path, names: dict[int, str] | None = None) -> BaseDetector:
    path = Path(weights)
    if not path.exists():
        raise FileNotFoundError(f"找不到权重: {path}")
    if path.suffix in {".onnx", ".engine", ".pt"}:
        # .pt 可能是 YOLO 也可能是我们自己的 checkpoint，用内容判断
        if path.suffix == ".pt":
            try:
                import torch

                checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and "meta" in checkpoint:
                    return TorchvisionDetector(path, names or {})
            except Exception:  # noqa: BLE001 - 不是我们的格式就当 YOLO 处理
                pass
        return YoloDetector(path)
    raise ValueError(f"不认识的权重格式: {path.suffix}")
