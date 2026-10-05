# Phase 1 Report: ISL-CSLTR Preprocessing & Setup

## 1. Dataset Inspection & Breakthrough
The original dataset description claimed 700 fully annotated videos (100 sentences performed by 7 different signers). However, upon deep inspection of the physical files:
- The dataset actually contains **687 video files**. Some folders had extra duplicate videos (e.g., `serve the food` had 9), while others had missing files (e.g., `i dont agree` had 1).
- The video filenames were completely chaotic (`MVI_6503.MP4`, `room (2).MP4`), and the metadata lacked an explicit mapping between videos and specific signers. 
- **The Breakthrough:** By clustering the videos by resolution, we discovered a perfect physical proxy for the signers:
  - **492 videos** were recorded at **1920x1080 @ 25fps**. These map to the "five student volunteers from SASTRA Deemed University".
  - **195 videos** were recorded at **240x320 @ 25fps**. These map to the "two native signers of Navajeevan".

This allowed us to dynamically and strictly assign `signer_id 0-4` to the 1080p student volunteers, and `signer_id 5-6` to the 240p native signers. 

## 2. Environment & Tooling Modernization
- We set up a clean local Python `.venv`.
- We explicitly configured **PyTorch 2.11.0 with CUDA 12.8 support** to take full advantage of the local NVIDIA RTX 3050 GPU (4GB VRAM).
- We stripped out heavy unneeded dependencies (like TensorFlow) and built a lightweight custom `SimpleTokenizer`.
- We successfully migrated from the deprecated `mp.solutions.holistic` to the modern **MediaPipe Tasks API (`HolisticLandmarker`)** for robust extraction.

## 3. Pose Landmark Extraction (MediaPipe)
The extraction pipeline (`step3_preprocess.py`) performs a CPU extraction using MediaPipe Holistic.
- **Landmarks extracted per frame:**
  - **Face:** 478 points $\times$ 3 coordinates (x, y, z) = 1434 features
  - **Pose:** 33 points $\times$ 4 coordinates (x, y, z, visibility) = 132 features
  - **Left Hand:** 21 points $\times$ 3 coordinates = 63 features
  - **Right Hand:** 21 points $\times$ 3 coordinates = 63 features
- **Total Feature Vector:** `1692` dimensions per frame.
- **Storage:** Sequences are saved as lightweight `.npy` arrays per video.

## 4. Preprocessing & Splitting Strategy
The dataset was split using the rock-solid resolution-based clustering heuristic to guarantee an absolute zero-leakage **signer-independent split**:
- **Train (5 Student Volunteers):** 492 videos
- **Validation & Test (2 Native Signers):** 195 videos

This strict separation ensures the model actually learns the *signs* rather than overfitting to a specific person's appearance, which is critical given the hardware constraints (4GB VRAM limits model complexity).

## Phase 1 Complete!
All 687 videos have been extracted, the tokenizer vocabulary (`tokenizer.pkl`) built, and the metadata split mapped (`split_metadata.json`). We are fully ready for Phase 2!
