# 容器化：TDDO CLI 的运行环境
# 数据、结果、密钥、许可证都不进镜像，运行时挂载。

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

RUN pip install --no-cache-dir . && \
    tddo --version

# 原始算例 83 MB、高德 Key、Gurobi 许可证均不写入镜像：
#   docker run --rm -v "%cd%/data:/app/data" --env-file .env tddo:latest tddo doctor
#   docker run --rm -v "%cd%/results:/app/results" tddo:latest tddo solve --set 104 --iters 300

ENTRYPOINT ["tddo"]
CMD ["--help"]
