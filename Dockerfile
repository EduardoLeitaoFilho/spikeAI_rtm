# SpikeAI: pose estimation (RTMPose) em CPU. Veja docs/DOCKER.md.
FROM python:3.12-slim
 
# libs de sistema exigidas pelo opencv-python (não-headless)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
 
COPY requirements.txt requirements-base.txt ./
RUN pip install --no-cache-dir -r requirements.txt
 
COPY run_pose.py ./
COPY src ./src
COPY config ./config
COPY docker ./docker
 
# Modelos ONNX ficam em /cache (volume): baixados na 1ª execução e reaproveitados nas seguintes.
ENV TORCH_HOME=/cache \
    PYTHONUNBUFFERED=1
RUN mkdir -p /cache /app/input /app/output && chmod 777 /cache /app/output
 
ENTRYPOINT ["python", "docker/processar.py"] 