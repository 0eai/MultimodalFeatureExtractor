#!/usr/bin/env bash

export PYTHONPATH=/home/ankit/Projects/MultimodalFeatureExtractor:$PYTHONPATH

# Text
# python3 scripts/main.py --task perception --modality transcriptions --hf_model 'SamLowe/roberta-base-go_emotions' --feature_segment go_emotions --step_size 0.5 --device cuda:0

# Audio
python3 scripts/main.py --task perception --modality wav --hf_model 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim' --feature_segment revdess --step_size 0.5 --window_size 2.0 --sampling_rate 16000 --device cuda:0

# Faces
# python3 scripts/main.py --task perception --modality faces --hf_model 'trpakov/vit-face-expression' --feature_segment fer --device cuda:0