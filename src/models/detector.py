"""改进的检测器：ResNet 骨干 + FPN 多尺度融合 + CBAM 注意力。

流程（对应论文里的"改进 CNN 模型层"）：

    输入 512x512
      -> ResNet50 从 C2/C3/C4/C5 抽多尺度特征
      -> FPN 自顶向下融合 + 3x3 平滑，统一到 256 通道
      -> CBAM 逐层加权（通道注意力 + 空间注意力）
      -> Faster R-CNN 的 RPN + ROIHead 输出 类别 + 框

为什么不是"从零搭一个网络"：检测头的正负样本分配、anchor 生成、NMS、损失平衡
这些工程细节和"改进点"无关，但写错一个就训不起来。用 torchvision 的标准检测头，
把改进集中在 backbone 上，实验结论才干净——这是能写进论文的取舍。

    from src.models.detector import build_detector
    model = build_detector(num_classes=7, use_cbam=True)
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from src.models.cbam import CBAM


class CBAMBackbone(nn.Module):
    """包一层 torchvision 的 BackboneWithFPN，在每层输出后插 CBAM。"""

    def __init__(self, inner: nn.Module, use_cbam: bool = True, reduction: int = 16) -> None:
        super().__init__()
        self.inner = inner
        self.out_channels = inner.out_channels
        self.use_cbam = use_cbam
        # FPN 输出固定 5 层（stride 4/8/16/32/64），按 out_channels 建注意力
        self.attention = (
            nn.ModuleList([CBAM(self.out_channels, reduction) for _ in range(5)])
            if use_cbam
            else None
        )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.inner(x)
        if self.attention is None:
            return features
        return OrderedDict(
            (name, module(feature))
            for (name, feature), module in zip(features.items(), self.attention, strict=True)
        )


def build_backbone(
    backbone: str = "resnet50",
    use_cbam: bool = True,
    trainable_layers: int = 5,
    pretrained: bool = True,
    reduction: int = 16,
) -> nn.Module:
    from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50
    from torchvision.models.detection.backbone_utils import BackboneWithFPN

    builders = {
        "resnet34": (resnet34, ResNet34_Weights.DEFAULT),
        "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    }
    if backbone not in builders:
        raise ValueError(f"只支持 {list(builders)}，收到 {backbone}")
    factory, weights = builders[backbone]
    model = factory(weights=weights if pretrained else None)

    # 取 C2~C5（layer1~layer4）用于 FPN；trainable_layers=5 表示全部参与微调
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [256, 512, 1024, 2048] if backbone == "resnet50" else [64, 128, 256, 512]
    from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

    fpn = BackboneWithFPN(
        model,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )

    # 冻结策略：从后往前保留 trainable_layers 个 stage
    if trainable_layers < 5:
        for name, parameter in fpn.body.named_parameters():
            if "layer4" in name:
                parameter.requires_grad = trainable_layers >= 1
            elif "layer3" in name:
                parameter.requires_grad = trainable_layers >= 2
            elif "layer2" in name:
                parameter.requires_grad = trainable_layers >= 3
            elif "layer1" in name:
                parameter.requires_grad = trainable_layers >= 4
            else:
                parameter.requires_grad = False

    return CBAMBackbone(fpn, use_cbam=use_cbam, reduction=reduction)


def build_detector(
    num_classes: int,
    backbone: str = "resnet50",
    use_cbam: bool = True,
    min_size: int = 512,
    max_size: int = 512,
    pretrained: bool = True,
) -> nn.Module:
    """num_classes 要包含背景（数据集 6 类 -> 传 7）。"""
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    size = (32, 64, 128, 256, 512)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(size)
    anchor_generator = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=aspect_ratios)

    model = FasterRCNN(
        backbone=build_backbone(
            backbone=backbone, use_cbam=use_cbam, pretrained=pretrained
        ),
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        rpn_anchor_generator=anchor_generator,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
