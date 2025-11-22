import logging
import os
import tempfile
from typing import List, Literal, Any, Dict, Optional

import torch
import whisper
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager

# --- Configurazione Iniziale ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = "cuda"
logging.info(f"Forzando l'utilizzo del device: {device}")

storage_client = storage.Client()

class PredictionRequest(BaseModel):
    instances: List[str] = Field(..., description="Lista di GCS URI degli audio da trascrivere.")
    parameters: Dict[str, Any] = Field(default_factory=dict)
   
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logging.info("Caricamento del modello Whisper 'large-v2' in corso...")
    app.state.model = whisper.load_model("large-v2", device=device, download_root='./')
    logging.info("Caricamento del modello Whisper 'large-v2' completato!")
    yield # Lifespan is completed    
    

# --- Applicazione FastAPI ---
app = FastAPI(title="Whisper Large V2 Transcription Service", lifespan=lifespan)

# --- Funzioni di Supporto (invariata) ---
def download_gcs_file(gcs_uri: str) -> str:    
    try:
        if not gcs_uri.startswith("gs://"):
            raise ValueError("URI non valido, deve iniziare con 'gs://'")
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        suffix = os.path.splitext(blob_name)[1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            blob.download_to_filename(tmp_file.name)
            logging.info(f"File scaricato da {gcs_uri} a {tmp_file.name}")
            return tmp_file.name
    except Exception as e:
        logging.error(f"Errore durante il download da GCS {gcs_uri}: {e}")
        raise HTTPException(status_code=500, detail=f"Impossibile scaricare il file: {gcs_uri}")

# --- Endpoint ---
@app.get("/health", status_code=200)
def health_check():
    # Il health check di Vertex AI deve restituire 200 se il server è pronto    
    return {"status": "healthy"}

@app.get("/readiness")
async def readiness_probe():
    return {"status": "alive"}

@app.get("/liveness")
async def liveness_probe():
    return {"status": "alive"}

@app.post("/predict")
def predict(prediction_request: PredictionRequest, request: Request):    

    predictions = []        
    
    lang_for_whisper = 'it'

    for instance_uri in prediction_request.instances:
        temp_audio_path = None
        try:
            temp_audio_path = download_gcs_file(instance_uri)            
            
            logging.info(f"Trascrizione di {instance_uri}")
            
            # Trascrivi l'audio
            model = request.app.state.model
            result = model.transcribe(temp_audio_path, language=lang_for_whisper, word_timestamps=True)            
                      
            logging.info(f"Trascrizione completata per {instance_uri}")
            
            segments = result['segments']            
            
            predictions.append({"result": segments})                            
            
        except Exception as e:
            logging.error(f"Errore durante la predizione per {instance_uri}: {e}", exc_info=True)
            predictions.append({"error": str(e), "instance": instance_uri})
            
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    return {"predictions": predictions}