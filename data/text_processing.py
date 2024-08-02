import torch
import pandas as pd
from datasets import Dataset
from pathlib import Path
import math
from moviepy.editor import VideoFileClip
from utils.video_utils import get_video_duration_ms

def preprocess(text):
    return " ".join(['@user' if t.startswith('@') and len(t) > 1 else 'http' if t.startswith('http') else t for t in text.split(" ")])

def get_text_dataset(args):
    files = [path.as_posix() for path in list(Path(args.paths['modality']).rglob("*.csv"))]
    files = sorted(files)
    return Dataset.from_dict({"csv": files})

def get_cls_representation(sentence, tokenizer, model, device):
    inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()

def process_subj_text(csv_path, tokenizer, model, args, column_names=None, feature_size=None):
    print(f'Currently extracting feature from text: {csv_path}')
    csv_path = Path(csv_path)
    file_name = csv_path.name
    subj_id = csv_path.stem
    out_file_path = Path(f'{args.paths["feature_segment"]}/{subj_id}.csv')
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    total_seg_t = get_video_duration_ms(f'{args.paths["vid_dir"]}/{subj_id}.mp4')
    txt_df = pd.read_csv(csv_path).dropna()
    features = {}
    for _, row in txt_df.iterrows():
        t_alpha, t_beta, sentence = row['start'] * 1000, row['end'] * 1000, row['sentence']
        if 'cardiffnlp/twitter-roberta-base-sentiment-latest' in model.name_or_path:
            sentence = preprocess(sentence)
        feature = get_cls_representation(sentence, tokenizer, model, args.device).cpu().numpy()
        t_a = int(math.floor(t_alpha / int(args.step_size * 1000))) * int(args.step_size * 1000)
        timestamps = list(range(t_a, int(t_beta), int(args.step_size * 1000)))
        if feature_size is None:
            feature_size = feature.size
        column_names = column_names = ['timestamp', 'segment_id'] + [str(i) for i in range(feature_size)]
        for timestamp in timestamps:
            _row = [timestamp, subj_id] + feature.tolist()
            features[timestamp] = pd.Series(_row, index=column_names)
    df = pd.DataFrame(columns=column_names)
    for timestamp in range(0, int(total_seg_t), int(args.step_size * 1000)):
        if timestamp in features:
            df.loc[len(df)] = features[timestamp]
        else:
            _row = [timestamp, subj_id] + [0] * feature_size
            df.loc[len(df)] = pd.Series(_row, index=column_names)
    df.to_csv(out_file_path, index=False)
    return column_names, feature_size

def process_text(data, model, tokenizer, args):
    feature_size = None
    column_names = None
    for csv_path in data['csv']:
        column_names, feature_size = process_subj_text(csv_path, tokenizer, model, args, column_names, feature_size)
