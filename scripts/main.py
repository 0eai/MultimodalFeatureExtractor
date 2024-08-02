import os
import sys
import argparse
import torch
from config.config import PATH_TO_FEATURES, TASKS, HUMOR_PATH, HUMOR, PERCEPTION_PATH, PERCEPTION, MODALITIES
from data.text_processing import process_text, get_text_dataset
from data.audio_processing import process_audio, get_audio_dataset
from data.image_processing import process_faces, get_faces_dataset
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, Wav2Vec2ForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(description='MuSe 2024.')
    parser.add_argument('--task', type=str, required=True, choices=TASKS, help=f'Specify the task from {TASKS}.')
    parser.add_argument('--modality', required=True, choices=MODALITIES, help='Specify the modality.')
    parser.add_argument('--hf_model', required=True, help='Specify HuggingFace model name.')
    parser.add_argument('--feature_segment', required=True, help='Specify the name of the feature segment used.')
    parser.add_argument('--step_size', type=float, default=0.5, help='Specify the step size in second (default: 0.5).')
    parser.add_argument('--window_size', type=float, default=2.0, help='Specify the window size in second (default: 2.0).')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Specify the sampling rate (default: 16000).')
    parser.add_argument('--device', type=str, default='cpu', help='Specify device')
    args = parser.parse_args()
    return args

def main(args):
    if args.task == HUMOR:
        pass
    elif args.task == PERCEPTION:
        if args.modality == 'faces':
            data = get_faces_dataset(args)
            processor = AutoImageProcessor.from_pretrained(args.hf_model)
            model = AutoModel.from_pretrained(args.hf_model)
            if 'cuda' in args.device:
                model.to(args.device)
            process_faces(data, model, processor, args)
        elif args.modality == 'wav':
            data = get_audio_dataset(args)
            processor = AutoFeatureExtractor.from_pretrained(args.hf_model)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(args.hf_model)
            if 'cuda' in args.device:
                model.to(args.device)
            process_audio(data, model, processor, args)
        elif args.modality == 'transcriptions':
            data = get_text_dataset(args)
            tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
            model = AutoModel.from_pretrained(args.hf_model)
            if 'cuda' in args.device:
                model.to(args.device)
            process_text(data, model, tokenizer, args)

if __name__ == '__main__':
    print("Start", flush=True)
    args = parse_args()
    args.paths = {
        'raw': os.path.join(HUMOR_PATH if args.task == HUMOR else PERCEPTION_PATH, 'raw_data' if args.task == HUMOR else 'raw'),
        # 'feature_segment': os.path.join(PATH_TO_FEATURES[args.task], args.feature_segment)}
        'feature_segment': os.path.join('feature_segments', args.feature_segment)
    }
    args.paths['modality'] = os.path.join(args.paths['raw'], args.modality)
    args.paths['vid_dir'] = os.path.join(args.paths['raw'], 'videos')
    print(args)
    main(args)
