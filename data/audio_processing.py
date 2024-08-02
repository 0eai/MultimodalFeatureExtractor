import torch
import pandas as pd
import numpy as np
from datasets import Dataset, Audio
from pathlib import Path
import math

def get_audio_dataset(args):
    files = [path.as_posix() for path in list(Path(args.paths['modality']).rglob("*.wav"))]
    files = sorted(files)
    return Dataset.from_dict({"audio": files}).cast_column("audio", Audio(sampling_rate=args.sampling_rate))

def get_processed_features(audio_data, processor, window_samples, step_samples, sampling_rate=16000):
    num_samples = len(audio_data["array"])
    features = []

    for i in range(0, num_samples, step_samples):
        features.append(audio_data["array"][i:i + window_samples])
    inputs = processor(features, sampling_rate=sampling_rate, padding='longest', max_length=window_samples, return_tensors="pt")
    return inputs

def process_subj_audio(audio_data, processor, model, window_samples, step_samples, args, column_names=None, feature_size=None):
    print(f'Currently extracting feature from audio: {audio_data["path"]}')
    subj_id = Path(audio_data['path']).stem
    out_file_path = Path(f'{args.paths["feature_segment"]}/{subj_id}.csv')
    out_file_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = get_processed_features(audio_data, processor, window_samples, step_samples, args.sampling_rate)

    if 'cuda' in args.device:
        inputs = {key: val.to(args.device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        feature = torch.mean(outputs.hidden_states[-1], dim=1)

    if feature_size is None:
        feature_size = feature.size(-1)
    column_names = ['timestamp', 'subj_id'] + ["feature_" + str(i + 1) for i in range(feature_size)]
    df = pd.DataFrame(columns=column_names)
    for i, val in enumerate(feature.cpu().numpy()):
        row = [str(i * int(args.step_size * 1000)), subj_id]
        row.extend(val)
        df.loc[len(df)] = row
    df.to_csv(out_file_path, index=False)
    return column_names, feature_size

def process_audio(data, model, processor, args):
    feature_size = None
    column_names = None
    window_samples = int(args.window_size * args.sampling_rate)
    step_samples = int(args.step_size * args.sampling_rate)
    for audio_data in data['audio']:
        column_names, feature_size = process_subj_audio(audio_data, processor, model, window_samples, step_samples, args, column_names, feature_size)
