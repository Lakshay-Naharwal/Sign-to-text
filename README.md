# ISL to English Translation Model

This project aims to build an Indian Sign Language (ISL) to English translation model using a Sequence-to-Sequence (Seq2Seq) architecture, specifically optimized to run locally on hardware with strict VRAM limits (4GB). The dataset used is the ISL-CSLRT Corpus, and signs are preprocessed into 1692-dimensional structural landmarks using MediaPipe (Holistic) to minimize computational overhead.

## Project Structure
- `data/raw/`: Original downloaded ISL-CSLTR videos.
- `data/processed/`: Extracted structural frame sequences as `.npy` files.
- `data/splits/`: The train/val/test mapping (`split_metadata.json`) and the built `tokenizer.pkl`.
- `src/preprocessing/`: Scripts to fetch, verify, and preprocess the dataset.
- `src/models/`: Neural network architecture definitions.
- `src/training/`: Training loop and data loading logic.
- `src/evaluation/`: Scripts to benchmark the model.
- `checkpoints/`: Model weights saved during training.
- `reports/`: Markdown reports and visual verifications of datasets and model progress.

## How to Reproduce

### Phase 1: Data Preprocessing
1. **Download Dataset**: Put the ISL-CSLRT Corpus into `data/raw/ISL_CSLRT_Corpus`. Alternatively, run `python src/preprocessing/fetch_dataset.py` if setting up from scratch.
2. **Extract Features**: We use a dynamic resolution-based split to prevent data leakage (1080p student volunteers for Train, 240p native signers for Val/Test).
   ```bash
   # Run the extraction pipeline
   python src/preprocessing/step3_preprocess.py
   ```
   This script will loop over the raw dataset and extract MediaPipe landmarks, saving them to `data/processed/*.npy`, and produce `data/splits/split_metadata.json`.

*(Additional phases will be documented here as they are completed.)*
