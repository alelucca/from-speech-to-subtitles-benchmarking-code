import subprocess
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from utils import names
from multiprocessing import Manager

# Funzioni di utilità
files = names.get_file_names()
models = names.get_model_names()
SCORES_FILE = "suber_score.txt"


def load_existing_scores(path=SCORES_FILE):
    """Carica le combinazioni già presenti nel file di score."""
    existing = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    model, file, _ = [x.strip() for x in line.split(" - ")]
                    existing.add((model, file))
                except ValueError:
                    continue  # in caso di righe malformate
    return existing


def clean_srt_file(path):
    """
    Legge e riscrive un file SRT, sostituendo i caratteri non validi
    con il simbolo � (errors='replace').
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File SRT non trovato: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    import unicodedata
    import re

    # nfkd = unicodedata.normalize('NFKD', content)
    # filtered = ''.join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[^\x00-\x7F]", "�", content)

    path = path.replace('.srt','') + '_decoded.srt'

    with open(path, "w", encoding="utf-8") as f:
        f.write(cleaned)
    
    return path


def get_suber(file, model, lock):
    """Esegue il calcolo dello score SubER tra due file SRT."""
    reference_srt_path = os.path.normpath(os.path.join("..", "data", "srt", "ground-truth-cleaned", f"{file}.srt"))
    hypothesis_srt_path = os.path.normpath(os.path.join("..", "data", model, "improved-srt", f"{file}.srt"))

    # Ripulisce il file ipotesi per evitare UnicodeDecodeError
    new_hypothesis_path = clean_srt_file(hypothesis_srt_path)

    command = [
        "suber",
        "-H", hypothesis_srt_path,
        "-R", reference_srt_path
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = result.stdout.strip()
    score = json.loads(output)

    print(f"[INFO] SubER score for file '{file}' with model '{model}': {score['SubER']}")

    with lock:
        with open(SCORES_FILE, "a", encoding="utf-8") as f:
            f.write(f"{model} - {file} - {score['SubER']}\n")

    return score['SubER']


def process_task(file, model, lock, existing):
    """Funzione wrapper per parallelizzare il calcolo SubER di un singolo file/model."""
    if (model, file) in existing:
        print(f"[SKIP] Già calcolato: file '{file}' con modello '{model}'")
        return (file, model, None)

    print(f"[INFO] Calcolo SubER per file '{file}' con modello '{model}'")
    return (file, model, get_suber(file, model, lock))


if __name__ == "__main__":
    # Imposta DataFrame
    suber_results = pd.DataFrame(index=files, columns=models)

    # Carica combinazioni già presenti
    existing = load_existing_scores()

    # Numero massimo di core disponibili
    max_workers = os.cpu_count() - 2

    tasks = [(file, model) for model in models for file in files]
    manager = Manager()
    lock = manager.Lock()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_task, file, model, lock, existing) for file, model in tasks]

        for future in as_completed(futures):
            try:
                file, model, score = future.result()
                if score is not None:
                    suber_results.loc[file, model] = score
            except Exception as e:
                print(f"[ERRORE] Durante l'elaborazione {file} {model}: {e}")

    print("\n=== Calcolo completato ===")
    #print(suber_results)
