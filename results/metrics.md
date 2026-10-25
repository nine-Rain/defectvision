| 方案 | mAP50-95 | mAP50 | Precision | Recall | 体积 (MB) | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| neu-yolo-test | 0.4041 | 0.7492 | 0.6967 | 0.6876 | 5.22 | yolo11n 50ep @320，NEU test |
| baseline-test | 0.3842 | 0.5590 | 0.6680 | 0.5297 | 5.24 | yolo11n 50ep @512，test |
| onnx-fp32 | 0.3759 | 0.5607 | 0.6990 | 0.5227 | 10.12 | ONNX fp32 |
| onnx-int8 | 0.3759 | 0.5602 | 0.6673 | 0.5321 | 2.88 | ONNX INT8 |
| seg-test | 0.3426 | 0.5802 | 0.7397 | 0.5200 | 5.75 | yolo11n-seg 50ep @512，test（掩码） |
| neu-cnn-test | 0.2990 | 0.6668 | — | 0.4787 | 158.31 | resnet50+FPN+CBAM，20ep@320，NEU test |
| cnn-nocbam-test | 0.2620 | 0.4935 | — | 0.3534 | 158.15 | resnet50+FPN，20ep@512，MSDD test |
| cnn-cbam-test | 0.2475 | 0.4755 | — | 0.3692 | 158.31 | resnet50+FPN+CBAM，20ep@512，MSDD test |
| smoke | 0.0930 | 0.1611 | 0.6760 | 0.2057 | 5.21 | 1 epoch 冒烟，仅验证流水线 |
