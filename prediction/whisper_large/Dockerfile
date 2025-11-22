# Dockerfile (Corretto e con git aggiunto)

# Usa un'immagine base NVIDIA con CUDA 11.8 e Python 3.10.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Imposta variabili d'ambiente per evitare interazioni durante l'installazione
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Installa dipendenze di sistema: Python, pip, ffmpeg e GIT
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Imposta la directory di lavoro
WORKDIR /app

# Copia il file dei requisiti prima del resto del codice
COPY requirements.txt .

COPY large-v2.pt .

# Installa le dipendenze Python usando l'extra-index-url specificato
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copia il codice dell'applicazione
COPY main.py .

# Esponi la porta che Vertex AI si aspetta
EXPOSE 8000

# Comando per avviare l'applicazione
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]