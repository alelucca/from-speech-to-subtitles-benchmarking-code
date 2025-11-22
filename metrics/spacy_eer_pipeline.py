import os
import json
import re
import pandas as pd
from rapidfuzz import fuzz
from standardization import standardization_utils

def process_gt_jsonl(gt_jsonl_path):
    """Carica le entità GT da un JSONL."""
    entities = []
    with open(gt_jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            data = json.loads(line.strip())
            for extr in data.get("extractions", []):
                entities.append({
                    "extraction_class": extr["extraction_class"],
                    "extraction_text": extr["extraction_text"],
                    "char_interval": extr["char_interval"],
                    "start_time": None,
                    "end_time": None
                })
    return entities

def timestamp_to_entities(subtitles, entities):
    """Allinea entità GT ai sottotitoli usando gli offset carattere."""
    offset = 0
    for sub in subtitles:
        sub_len = len(sub.text)
        for entity in entities:
            start_char, end_char = entity["char_interval"]
            if start_char >= offset and end_char <= offset + sub_len:
                entity["start_time"] = sub.start_time
                entity["end_time"] = sub.end_time
        offset += sub_len + 1  # +1 per spazio/line break
    return entities

def clean_entity_text(text):
    """Normalizza il testo rimuovendo punteggiatura e spazi extra."""
    if "'" in text:
        text = text.split("'", 1)[1]
    return re.sub(r"[^\w']+", " ", text).strip().lower()

def match_entities(gt_entities, asr_segments, threshold=0.9, time_pad=500):
    """Trova il miglior match per ogni entità GT tra i segmenti ASR."""
    results = []
    asr_segments = sorted(asr_segments, key=lambda s: s.start_time)

    for ent in gt_entities:
        gt_text = clean_entity_text(ent["extraction_text"])
        gt_start = ent["start_time"]
        gt_end = ent["end_time"]

        if gt_start is None or gt_end is None:
            results.append("")
            continue

        win_lo = gt_start - time_pad
        win_hi = gt_end + time_pad

        merged_words = []
        for seg in asr_segments:
            if seg.end_time < win_lo or seg.start_time > win_hi:
                continue
            merged_words.extend(seg.text.split())

        if not merged_words:
            results.append("")
            continue

        n_tokens = len(gt_text.split())
        ngrams = (
            [" ".join(merged_words[i:i+n_tokens]) for i in range(len(merged_words)-n_tokens+1)]
            if n_tokens > 1 else merged_words
        )

        best_ngram = ""
        best_sim = 0.0
        for ng in ngrams:
            sim = fuzz.ratio(gt_text, clean_entity_text(ng)) / 100.0
            if sim > best_sim:
                best_sim = sim
                best_ngram = ng

        best_ngram = best_ngram if best_sim >= threshold else ""

        results.append(best_ngram)

    return results

def compare_multiple_asr(files, models, threshold=0.9, time_pad=5):
    """
    Confronta le entità di più file GT con più sistemi ASR.
    Ritorna un DataFrame con colonne: program, gt_entity, extraction_class, modello1, modello2...
    """
    all_results = []

    for file in files:
        gt_jsonl_path = f"../data/jsonl_spacy/{file}.jsonl"
        gt_srt_path = f"../data/srt/ground-truth-cleaned/{file}.srt"

        with open(gt_srt_path, "r", encoding="utf-8", errors="replace") as f:
                srt_text = f.read()

        gt_subtitles = standardization_utils.preprocess(srt_text)
        # Carico GT
        gt_entities = process_gt_jsonl(gt_jsonl_path)        
        gt_timestamp_entities = timestamp_to_entities(gt_subtitles, gt_entities)

        gt_timestamp_entities = [e for e in gt_timestamp_entities if e.get("start_time") is not None and e.get("end_time") is not None]

        start_time_sub = [round(float(e["start_time"])/1000, 3) for e in gt_timestamp_entities]
        end_time_sub   = [round(float(e["end_time"])/1000, 3) for e in gt_timestamp_entities]
        win_time_start = [sts - float(time_pad) for sts in start_time_sub]
        win_time_end = [ets + float(time_pad) for ets in end_time_sub]

        # Preparo la riga iniziale
        result_row = {
            "program": file,
            "gt_entity": [e["extraction_text"] for e in gt_timestamp_entities],
            "start_time_sub": start_time_sub,
            "end_time_sub": end_time_sub,
            "extraction_class": [e["extraction_class"] for e in gt_timestamp_entities],
            "win_time_start": win_time_start,
            "win_time_end": win_time_end 
        }

        # Per ogni modello, dividi i risultati in 3 colonne
        for model in models:
            asr_path = f"../data/{model}/srt/{file}.srt"
            with open(asr_path, "r", encoding="utf-8", errors="replace") as f:
                srt_text = f.read()
            asr_subtitles = standardization_utils.preprocess(srt_text)
            matches = match_entities(gt_timestamp_entities, asr_subtitles, threshold, time_pad)
            # matches è una lista di tuple (best_ngram, win_lo, win_hi)
            result_row[model] = matches           

        df_partial = pd.DataFrame(result_row)
        all_results.append(df_partial)

    return pd.concat(all_results, ignore_index=True)
