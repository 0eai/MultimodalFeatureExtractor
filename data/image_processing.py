import torch
import pandas as pd
from datasets import Dataset, Image
from pathlib import Path
import time
from utils.file_utils import natural_sort_key

def get_all_files_sorted(folder_path, file_extension="*"):
    folder = Path(folder_path)
    files = [path.as_posix() for path in list(folder.glob(file_extension))]
    sorted_files = sorted(files, key=lambda f: natural_sort_key(f.split('/')[-1]))
    return sorted_files

def get_subj_files(directories):
    subj_files = {}
    for dir in directories:
        dir_name = Path(dir).name
        subj_files[dir_name] = get_all_files_sorted(dir, "*.jpg")
    return subj_files

def get_all_datasets(subj_files):
    img_datasets = {}
    for subj_id, files in subj_files.items():
        img_dataset = Dataset.from_dict({
            "image": files,
            "path": files,
            "dir_name": [subj_id] * len(files)
        }).cast_column("image", Image())
        img_datasets[subj_id] = img_dataset
    return img_datasets

def get_faces_dataset(args):
    directories = [d for d in Path(args.paths['modality']).rglob('*') if d.is_dir()]
    directories.sort(key=lambda x: int(x.name))
    subj_files = get_subj_files(directories)
    return get_all_datasets(subj_files)

def extract_info(file_path):
    path_parts = file_path.split('/')
    subj_id = int(path_parts[-2])
    timestamp = int(path_parts[-1].split('_')[1])
    return subj_id, timestamp

def process_subj_faces(subj_id, subj_dataset, processor, model, args, column_names=None, feature_size=None):
    print(f'Starting to extract features from subject {subj_id} images')
    out_file_path = Path(f'{args.paths["feature_segment"]}/{subj_id}.csv')
    out_file_path.parent.mkdir(parents=True, exist_ok=True)

    for index, img_data in enumerate(subj_dataset):
        start_time = time.time()
        print(f'\tCurrently extracting feature from image: {img_data["path"]}')
        _, timestamp = extract_info(img_data["path"])

        inputs = processor(img_data['image'], return_tensors="pt")

        if 'cuda' in args.device:
            inputs = {key: val.to(args.device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_layer = outputs.last_hidden_state
            feature = last_hidden_layer[:, 0, :]
            out_feature = feature.cpu().numpy()

        if feature_size is None:
            feature_size = feature.size(-1)
        column_names = ['timestamp', 'subj_id'] + [str(i) for i in range(feature_size)]
        if index == 0:
            df = pd.DataFrame(columns=column_names)

        row = [timestamp, subj_id]
        row += out_feature.tolist()[0]
        df.loc[len(df)] = row
    df.to_csv(out_file_path, index=False)
    return column_names, feature_size

def process_faces(data, model, processor, args):
    feature_size = None
    column_names = None
    for subj_id, subj_dataset in data.items():
        column_names, feature_size = process_subj_faces(subj_id, subj_dataset, processor, model, args, column_names, feature_size)
