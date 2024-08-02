import os
from pathlib import Path

# adjust your paths here.
BASE_PATH = "/mnt/sda/datasets/MuSe/2024/"
PERCEPTION_PATH = os.path.join(BASE_PATH, 'c1_muse_perception')
HUMOR_PATH = os.path.join(BASE_PATH, 'c2_muse_humor')

HUMOR = 'humor'
PERCEPTION = 'perception'

PATH_TO_FEATURES = {
    PERCEPTION: os.path.join(PERCEPTION_PATH, 'feature_segments'),
    HUMOR: os.path.join(HUMOR_PATH, 'feature_segments')
}

PATH_TO_LABELS = {
    PERCEPTION: os.path.join(PERCEPTION_PATH, 'labels.csv'),
    HUMOR: os.path.join(HUMOR_PATH, 'label_segments')
}

PATH_TO_METADATA = {
    PERCEPTION: os.path.join(PERCEPTION_PATH, 'metadata'),
    HUMOR: os.path.join(HUMOR_PATH, 'metadata')
}

TASKS = [PERCEPTION, HUMOR]  # Update with actual tasks

MODALITIES = ['audio', 'wav', 'videos', 'faces', 'texts', 'transcriptions']
