# LSTM-based Pose Classification Model Usage Guide

## Project Description
This project utilizes an LSTM model to classify human poses from skeleton data extracted using Mediapipe. The project consists of three main notebooks:

1. **create_data.ipynb**: Generates training data for the LSTM model.
2. **train_model.py**: Trains the LSTM model for pose classification.
3. **recog_lstm.py**: Real-time pose recognition from webcam.

## Environment Setup
Required libraries:
```
pip install -r requirements.txt
```

## Usage Guide
### Step 1: Data Generation
Run `create_data.ipynb` to generate training data:
- The notebook uses Mediapipe to extract skeleton data from video or image sources.
- The generated data will be saved in CSV format for model training.
- The pose labels include:
  - ngoi_lam_viec
  - ngoi_nga_lung
  - nam_ngu
  - gac_chan
  - dung_day
  - di_lai

### Step 2: Model Training
Run `train_model.py` to train the model:
- Ensure the CSV files are correctly located.
- The trained model will be automatically saved as `best_lstm_model.keras`.

### Step 3: Real-Time Pose Recognition
Run `recog_lstm.py`:
- You need to download this "!wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" or download by powershell " Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" -OutFile "pose_landmarker_heavy.task" " (make sure you are in the folder with the recog_lstm.py file in powershell)
- The webcam will be activated for pose recognition.
- Press `q` to exit.

# Human-Pose_recognition
