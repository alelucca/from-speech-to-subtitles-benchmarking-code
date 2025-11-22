import json
import os
import re
import pandas as pd

class Subtitle:
    def __init__(self, start_time: str, end_time: str, text: str):        
        self.start_time = start_time
        self.end_time = end_time
        self.text = text

def convert_str_to_ms(time: str):
    # time format: HH:MM:SS,mmm
    h, m, s_ms = time.split(':')
    s, ms = s_ms.split(',')
    return (int(h)*3600 + int(m)*60 + int(s))*1000 + int(ms)

def preprocess(srt_text):    
    pattern = re.compile(r'(\d+)\s+([\d:,]+) --> ([\d:,]+)\s+([\s\S]*?)(?=\n\d+\n|\Z)', re.MULTILINE)
    subtitles = []    
    for match in pattern.finditer(srt_text):
        start_time = convert_str_to_ms(match.group(2).strip())
        end_time = convert_str_to_ms(match.group(3).strip())
        text = match.group(4).strip()
        if text:               
            subtitles.append(Subtitle(start_time, end_time, text))            
    return subtitles

def load_all_subtitles(folders: list):
    all_subtitles = []
    for folder in folders:        
        for filename in os.listdir(folder):            
            with open(f"{folder}/{filename}", 'r', encoding='utf-8') as f:
                srt_content = f.read()
                subtitles = preprocess(srt_content)
                all_subtitles.append((folder.split('/')[2], filename, subtitles))
    
    return all_subtitles

def build_statistics_dataset(all_subtitles):    
    # Ricava tutti i nomi dei programmi e dei modelli
    program_names = sorted(set(fn.replace(".srt", "") for _, fn, _ in all_subtitles))
    model_names = sorted(set(model for model, _, _ in all_subtitles))

    # Crea dizionario per raccogliere i dati
    stats = {prog: {} for prog in program_names}

    with open("program_duration.json", "r", encoding="utf-8") as f:
        duration_data = json.load(f)

    for model_name, filename, subtitles in all_subtitles:
        prog = filename.replace(".srt", "")       
        
        durata_min = duration_data[prog] / 60           

        num_segments = len(subtitles)
        if num_segments == 0:
            cps = 0
            mean_segment_duration = 0
        else:
            total_duration_ms = subtitles[-1].end_time - subtitles[0].start_time
            total_duration_sec = total_duration_ms / 1000 if total_duration_ms > 0 else 1
            num_chars = sum(len(sub.text) for sub in subtitles)
            cps = num_chars / total_duration_sec if total_duration_sec > 0 else 0
            mean_segment_duration = (
                sum((sub.end_time - sub.start_time) for sub in subtitles) / num_segments / 1000
            )
        stats[prog]["DURATION"] = round(durata_min, 2)
        stats[prog][f"{model_name}_NUM_SEGMENTS"] = round(num_segments, 2)
        stats[prog][f"{model_name}_NUM_CHARS_SEGMENTS"] = round(num_chars/num_segments, 2)
        stats[prog][f"{model_name}_CPS"] = round(cps, 2)
        stats[prog][f"{model_name}_MEAN_SEGMENT_DURATION"] = round(mean_segment_duration, 2)

    # Ordina le colonne: prima tutti i CPS, poi tutte le MEAN SEGMENT DURATION
    ordered_columns = ["DURATION"] + [f"{model}_NUM_SEGMENTS" for model in model_names] + [f"{model}_CPS" for model in model_names] + [f"{model}_MEAN_SEGMENT_DURATION" for model in model_names] + [f"{model}_NUM_CHARS_SEGMENTS" for model in model_names]
    df = pd.DataFrame.from_dict(stats, orient="index")
    df = df.reindex(columns=ordered_columns)
    return df
