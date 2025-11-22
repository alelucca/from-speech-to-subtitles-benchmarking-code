## From Speech to Subtitles: related benchmarking code

This repository contains the code to benchmark four ASR systems (Whisper Large v2, WhisperX, AssemblyAI Universal, and NVIDIA Parakeet 0.6b V3) on a 50‑hour dataset of Italian TV episodes.

The original data are proprietary and not public, so they are **not** available.

### Folder overview:
- prediction: code to run predictions with each model. (Whisper Large, WhisperX, and Parakeet predictions were executed on a Vertex AI Workbench instance in Google Cloud Platform, using Cloud Storage for audio processing.)
- standardization: code to standardize ground truth and ASR predictions.
- metrics: notebooks for computing and exploring metrics.
- reviewer_llm: code to implement subtitle correction by the two LLM reviewers used in the study.
- raw_results: raw results from metric calculations.

### Abstract:
The integration of subtitles in video content is today an essential element for enhancing accessibility and audience engagement, extending beyond individuals with hearing impairments. Modern Automatic Speech Recognition (ASR) systems, based on Encoder-Decoder neural network architectures and trained on vast datasets, have progressively reduced transcription errors on standard benchmark datasets. However, their performance in real-world production scenarios, particularly for the subtitling of long-form Italian-language videos, remains largely unexplored.

This research aims to evaluate four state-of-the-art ASR models, Whisper Large v2, AssemblyAI Universal, Parakeet TDT v3 0.6b, and WhisperX, using a 40-hour dataset comprising 30 subtitled Italian television episodes produced. The study highlights both the strengths and limitations of these models, benchmarking their performance against professional human subtitlers. The findings indicate that these models are not yet ready for fully autonomous use at the accuracy level required in the Media Industry, but they can serve as highly effective tools to boost human productivity, with human-in-the-loop setups remaining crucial for achieving the best results.




