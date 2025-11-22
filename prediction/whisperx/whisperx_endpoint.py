import gc
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import torch
import whisperx
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = "cuda"
BATCH_SIZE = 16
COMPUTE_TYPE = "float16"
LANGUAGE = "it"

storage_client = storage.Client()

class PredictionRequest(BaseModel):
    instances: List[str] = Field(..., description="Lista di GCS URI degli audio da trascrivere.")
    parameters: Dict[str, Any] = Field(default_factory=dict)


def convert_to_json_serializable(data: Any) -> Any:
    """
    Converte ricorsivamente i tipi di dato non serializzabili (es. NumPy) in tipi nativi Python.
    """
    if isinstance(data, dict):
        return {k: convert_to_json_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_json_serializable(i) for i in data]
    if hasattr(data, 'item'):  # Heuristica per i tipi NumPy (es. np.float64)
        return data.item()
    return data

def download_gcs_file(gcs_uri: str) -> str:
    # (Funzione invariata, era già corretta)
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


# --- Gestione del Ciclo di Vita dell'Applicazione ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Inizio caricamento modelli...")
    app.state.whisper_model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)
    logging.info("Caricamento del modello Whisper 'large-v2' completato!")
    
    app.state.model_a, app.state.metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)
    logging.info("Caricamento del modello di allineamento completato!")    
    
    yield
    
    # Pulizia
    logging.info("Rilascio risorse dei modelli...")
    del app.state.whisper_model
    del app.state.model_a
    del app.state.metadata
    del app.state.diarization_model
    gc.collect()
    torch.cuda.empty_cache()

app = FastAPI(title="WhisperX Transcription Service", lifespan=lifespan)


# --- Endpoint ---
@app.get("/health", status_code=200)
def health_check():
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
    
    for uri in prediction_request.instances:
        temp_audio_path = None
        try:
            temp_audio_path = download_gcs_file(uri)
            audio = whisperx.load_audio(temp_audio_path)
            logging.info(f"Trascrizione di {uri} in corso...")

            # --- Pipeline di trascrizione ---
            
            # 1. Trascrizione            
            whisper_model = request.app.state.whisper_model
            transcription = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)
            logging.info(f"Trascrizione iniziale completata per {uri}")
            
            segments = transcription['segments']            
            
            # 3. Allineamento
            model_a = request.app.state.model_a
            metadata = request.app.state.metadata
            aligned_transcription = whisperx.align(segments, model_a, metadata, audio, DEVICE, return_char_alignments=False)
            logging.info(f"Allineamento completato per {uri}")  
            
            
            # 5. Converti in tipi JSON-serializzabili
            final_result = convert_to_json_serializable(aligned_transcription['segments'])
            
            # 6. Risultato finale
            predictions.append({"result": final_result})

        except Exception as e:
            logging.error(f"Errore durante la predizione per {uri}: {e}", exc_info=True)
            # Restituisce un errore specifico per l'istanza che ha fallito
            predictions.append({"error": str(e), "instance": uri})
        finally:
            # Pulizia del file temporaneo e della cache CUDA per il prossimo ciclo
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            gc.collect()
            torch.cuda.empty_cache()

    return {"predictions": predictions}