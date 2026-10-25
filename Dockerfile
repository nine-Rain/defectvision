# Demo 镜像：只装推理 + 界面，不含训练依赖。
# 构建: docker build -t defectvision .
# 运行: docker run -p 7860:7860 -v "$PWD/models:/app/models" defectvision
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 依赖层先拷，改代码不会导致重装依赖
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[train,demo]"

COPY configs ./configs

# 权重不打进镜像，运行时挂载 models/ 目录
VOLUME ["/app/models"]
EXPOSE 7860

CMD ["python", "-m", "src.app", "--weights", "models/best.onnx", "--host", "0.0.0.0"]
