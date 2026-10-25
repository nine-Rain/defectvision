# 模型产物目录（不进 git）

`src.export` 会把导出的 ONNX / INT8 模型统一放到这里：

```
models/
├── best.onnx
└── best_int8.onnx
```

部署时挂载这个目录，或者从 Release 里下载。
