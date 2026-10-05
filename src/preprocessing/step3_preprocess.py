import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
import glob

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<OOV>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<OOV>', 2: '<START>', 3: '<END>'}
        self.vocab_size = 4

    def fit_on_texts(self, texts):
        words = []
        for text in texts:
            words.extend(text.lower().split())
        counts = Counter(words)
        for word, _ in counts.most_common():
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def texts_to_sequences(self, texts):
        return [[self.word2idx.get(w, 1) for w in text.lower().split()] for text in texts]

# Initialize MediaPipe Holistic Tasks API
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to downloaded task model
MODEL_PATH = 'holistic_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task'

import urllib.request
if not os.path.exists(MODEL_PATH):
    print(f"Downloading MediaPipe model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

def extract_landmarks(results):
    """Extract and flatten landmarks from mediapipe Tasks API results."""
    # Face (478 landmarks * 3 coords = 1434)
    if results.face_landmarks and len(results.face_landmarks) > 0:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks]).flatten()
    else:
        face = np.zeros(1434)
        
    # Pose (33 landmarks * 4 coords = 132) - visibility and presence included in NormalizedLandmark
    if results.pose_landmarks and len(results.pose_landmarks) > 0:
        pose = np.array([[res.x, res.y, res.z, getattr(res, 'visibility', 0.0)] for res in results.pose_landmarks]).flatten()
    else:
        pose = np.zeros(132)
        
    # Left Hand (21 landmarks * 3 coords = 63)
    if results.left_hand_landmarks and len(results.left_hand_landmarks) > 0:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks]).flatten()
    else:
        lh = np.zeros(63)
        
    # Right Hand (21 landmarks * 3 coords = 63)
    if results.right_hand_landmarks and len(results.right_hand_landmarks) > 0:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks]).flatten()
    else:
        rh = np.zeros(63)
        
    return np.concatenate([pose, face, lh, rh])

def normalize_landmarks(landmarks):
    """Normalize landmarks (e.g. standard scaling or min-max per frame, or just keep as is since coordinates are already normalized 0-1 by MediaPipe)"""
    return landmarks

def process_video(video_path):
    """Process a single video and return sequence of landmarks."""
    cap = cv2.VideoCapture(video_path)
    sequence = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
        
    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO
    )
    
    with HolisticLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            # Calculate timestamp in ms
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            # Make detection
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Extract and normalize
            landmarks = extract_landmarks(results)
            landmarks = normalize_landmarks(landmarks)
            sequence.append(landmarks)
            
            frame_idx += 1
            
    cap.release()
    return np.array(sequence)

def main():
    DATASET_DIR = "data/raw/ISL_CSLRT_Corpus" # Adjust based on actual extracted folder name
    OUTPUT_PROCESSED_DIR = "data/processed"
    OUTPUT_SPLITS_DIR = "data/splits"
    
    os.makedirs(OUTPUT_PROCESSED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_SPLITS_DIR, exist_ok=True)
        
    # 1. Discover dataset structure
    # The dataset contains folders for each sentence under Videos_Sentence_Level.
    video_files = []
    for ext in ["*.mp4", "*.MP4", "*.avi", "*.AVI"]:
        video_files.extend(glob.glob(os.path.join(DATASET_DIR, "Videos_Sentence_Level", "**", ext), recursive=True))
        
    # Deduplicate files (glob is case-insensitive on Windows, causing double-counting)
    video_files = list(set(video_files))
        
    print(f"Found {len(video_files)} video files.")
    
    # Group videos by sentence (folder name)
    from collections import defaultdict
    sentence_to_videos = defaultdict(list)
    for vp in video_files:
        sentence = os.path.basename(os.path.dirname(vp))
        sentence_to_videos[sentence].append(vp)
    
    dataset_metadata = []
    
    # 2. Extract Landmarks
    print("Starting MediaPipe pose extraction...")
    total_videos = len(video_files)
    processed_count = 0
    
    for sentence, vps in sentence_to_videos.items():
        # Sort video paths alphabetically
        vps = sorted(vps)
        
        # Categorize videos based on resolution to separate signers
        # 1920x1080 -> Student volunteers (Train)
        # 240x320 -> Native signers (Val/Test)
        students = []
        natives = []
        
        for video_path in vps:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                if w > 1000:
                    students.append(video_path)
                else:
                    natives.append(video_path)
            cap.release()
            
        # Assign signer IDs: 0..4 for students, 5..6 for natives
        # Even if counts aren't exactly 5 and 2, this cleanly partitions them.
        mapped_videos = []
        for i, vp in enumerate(students):
            mapped_videos.append((i, vp))
        for i, vp in enumerate(natives):
            mapped_videos.append((5 + i, vp))
            
        for signer_id, video_path in mapped_videos:
            processed_count += 1
            print(f"Processing {processed_count}/{total_videos}: {video_path}")
            
            seq_name = f"{sentence.replace(' ', '_')}_signer_{signer_id}"
            out_path = os.path.join(OUTPUT_PROCESSED_DIR, f"{seq_name}.npy")
            
            # Extract and save if not already done
            if not os.path.exists(out_path):
                seq = process_video(video_path)
                np.save(out_path, seq)
            
            dataset_metadata.append({
                'video_path': video_path,
                'sentence': sentence,
                'signer_id': signer_id,
                'seq_name': seq_name,
                'npy_path': out_path
            })
            
    # 3. Tokenizer
    print("Building English tokenizer...")
    sentences = [item['sentence'] for item in dataset_metadata]
    tokenizer = SimpleTokenizer()
    tokenizer.fit_on_texts(sentences)
    with open(os.path.join(OUTPUT_SPLITS_DIR, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
        
    # 4. Signer-Independent Split
    print("Creating strict resolution-based signer-independent split...")
    
    train_data = []
    val_data = []
    test_data = []

    # Group by sentence to determine Val/Test consistently
    from collections import defaultdict
    sentence_to_items = defaultdict(list)
    for item in dataset_metadata:
        sentence_to_items[item['sentence']].append(item)

    for sentence, items in sentence_to_items.items():
        natives = []
        for item in items:
            cap = cv2.VideoCapture(item['video_path'])
            if cap.isOpened():
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                if w > 1000:
                    train_data.append(item)
                else:
                    natives.append(item)
            cap.release()
            
        natives = sorted(natives, key=lambda x: x['video_path'])
        if len(natives) >= 1:
            val_data.append(natives[0])
        if len(natives) >= 2:
            test_data.append(natives[1])
        if len(natives) > 2:
            test_data.extend(natives[2:])
            
    with open(os.path.join(OUTPUT_SPLITS_DIR, 'split_metadata.json'), 'w') as f:
        json.dump({
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'train_count': len(train_data),
            'val_count': len(val_data),
            'test_count': len(test_data)
        }, f, indent=4)
    
    print(f"Preprocessing complete. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
if __name__ == "__main__":
    main()
