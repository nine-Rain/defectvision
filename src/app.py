"""本地 Demo：上传一张图，看检出结果和耗时。

    python -m src.app
    python -m src.app --weights models/best.onnx

一个能点的界面，比一段训练日志有说服力得多——README 里那张 GIF 就来自这里。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.config import repo_path


def build_demo(weights: Path, default_conf: float = 0.25):
    try:
        import gradio as gr
        import numpy as np
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - 环境问题，不是逻辑问题
        raise SystemExit(
            f"缺少依赖: {exc}\n请先安装: pip install -e \".[train,demo]\""
        ) from exc

    model = YOLO(str(weights))

    def predict(image: np.ndarray | None, conf: float):
        if image is None:
            return None, {}, "请先上传一张图片"
        # Gradio 给的是 RGB，Ultralytics 吃 BGR
        bgr = np.ascontiguousarray(image[:, :, ::-1])
        start = time.perf_counter()
        result = model.predict(bgr, conf=float(conf), verbose=False)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        annotated = np.ascontiguousarray(result.plot()[:, :, ::-1])
        names = result.names
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                detections.append(
                    {
                        "class": names[int(box.cls.item())],
                        "confidence": round(float(box.conf.item()), 4),
                        "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                    }
                )
        summary = f"检出 {len(detections)} 个目标，耗时 {elapsed_ms:.1f} ms"
        return annotated, {"detections": detections}, summary

    with gr.Blocks(title="DefectVision") as demo:
        gr.Markdown(
            "## DefectVision\n上传一张待检图片，模型会框出缺陷并给出置信度与耗时。"
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="输入图片")
                conf_slider = gr.Slider(0.05, 0.95, value=default_conf, step=0.05, label="置信度阈值")
                run_button = gr.Button("检测", variant="primary")
            with gr.Column():
                image_output = gr.Image(type="numpy", label="检测结果")
                json_output = gr.JSON(label="结构化结果")
                text_output = gr.Textbox(label="摘要")
        run_button.click(predict, [image_input, conf_slider], [image_output, json_output, text_output])
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.app", description="启动 Gradio Demo")
    parser.add_argument("--weights", default="runs/baseline/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="生成公网临时链接（演示给别人看）")
    args = parser.parse_args(argv)

    weights = repo_path(args.weights)
    if not weights.exists():
        raise SystemExit(
            f"找不到权重: {weights}\n先训练: python -m src.train --smoke"
        )

    demo = build_demo(weights, args.conf)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
