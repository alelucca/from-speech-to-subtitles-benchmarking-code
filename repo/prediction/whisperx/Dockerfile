FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \    
    && rm -rf /var/lib/apt/lists/*


ENV XDG_CACHE_HOME=/app/cache
ENV HF_HOME=/app/cache/huggingface
ENV TORCH_HOME=/app/cache/torch

WORKDIR /app

COPY cache /app/cache
COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]