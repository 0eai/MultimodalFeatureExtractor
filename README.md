# MultimodalFeatureExtractor

## Overview

This project is designed to process multimodal data for feature extraction. It includes modules for handling text, audio, and image data, extracting features using pretrained HuggingFace models, and organizing the data for further analysis.

## Directory Structure

- `config/`: Configuration files and constants.
- `data/`: Data processing scripts for different modalities.
- `utils/`: Utility functions for file handling and video processing.
- `scripts/`: Main script to run the processing pipeline.
- `requirements.txt`: Python dependencies.
- `README.md`: Project overview and instructions.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/0eai/MultimodalFeatureExtractor.git
    cd MultimodalFeatureExtractor
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the main script, use the following command:
```sh
python scripts/main.py --task <TASK> --modality <MODALITY> --hf_model <MODEL_NAME> --feature_segment <FEATURE_SEGMENT> --step_size <STEP_SIZE> --window_size <WINDOW_SIZE> --sampling_rate <SAMPLING_RATE> --device <DEVICE>
