# 🤟 Gesture ↔ Speech Engine

A bidirectional communication system that translates hand gestures to speech and vice versa.

## Features

- **Gesture → Speech**: Real-time hand gesture recognition with text-to-speech output
- **Speech → Gesture**: Look up gestures by name in the vocabulary
- **Custom Training**: Collect your own gesture data and train custom classifiers
- **Web Interface**: Streamlit app with easy-to-use tabs

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Camera    │───▶│  MediaPipe  │───▶│  TensorFlow │
│   (OpenCV)  │    │   Hands     │    │   Keras     │
└─────────────┘    └─────────────┘    └─────────────┘
                          │                  │
                          ▼                  ▼
                   63 Landmarks        Prediction
                   (21 × 3)           + Confidence
                          │                  │
                          └────────┬─────────┘
                                   ▼
                          ┌─────────────┐
                          │   pyttsx3   │
                          │    (TTS)    │
                          └─────────────┘
```

## Installation

### Prerequisites
- Python 3.10+
- Webcam

### Setup

```bash
# Clone or navigate to project
cd "gesture project"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**Tabs:**
- **Collect Data**: Record gesture samples with your webcam
- **Train Model**: Preprocess data and train the classifier
- **Live Demo**: Real-time gesture recognition with speech

### 2. Command Line Tools

#### Collect Gesture Data
```bash
python collect_data.py --gesture HELLO --samples 50
python collect_data.py --gesture THUMBS_UP --samples 50
```

#### Preprocess Data
```bash
python preprocess.py --input data/raw --output data/processed --augment
```

#### Train Model
```bash
python train.py --data data/processed --output models/gesture_model.h5
```

#### Run Live Inference
```bash
python inference.py --model models/gesture_model.h5
```

## Project Structure

```
gesture project/
├── app.py              # Streamlit web interface
├── collect_data.py     # Data collection script
├── preprocess.py       # Data preprocessing & augmentation
├── model.py            # MLP model definition
├── train.py            # Training script
├── inference.py        # Live inference with TTS
├── utils.py            # Helper functions
├── export_gestures.py  # Export gestures to JSON
├── gestures.json       # Gesture vocabulary
├── requirements.txt    # Python dependencies
├── data/
│   ├── raw/            # Raw CSV files
│   └── processed/      # Preprocessed numpy arrays
├── models/
│   ├── gesture_model.h5    # Trained model
│   └── label_encoder.pkl   # Label encoder
└── sample_dataset/
    └── README.md       # Sample data documentation
```

## Gesture Data Format

### CSV Format (Raw Data)
```csv
lm_0,lm_1,lm_2,...,lm_62,label,timestamp
0.0,0.0,0.0,...,0.15,HELLO,2024-01-01T12:00:00
```

### gestures.json Format
```json
{
  "HELLO": {
    "description": "Wave hand gesture for greeting",
    "examples": [[0.0, 0.0, 0.0, ..., 0.15], ...]
  }
}
```

## Model Architecture

MLP (Multi-Layer Perceptron) Classifier:
- Input: 63 features (21 landmarks × 3 coordinates)
- Hidden: 128 → 64 → 32 neurons with ReLU + BatchNorm + Dropout
- Output: Softmax over N gesture classes

## Controls (Live Demo)

| Key | Action |
|-----|--------|
| Q / ESC | Quit |
| S | Toggle speech |
| +/- | Adjust confidence threshold |

## Adding New Gestures

1. **Collect samples**: Use `collect_data.py` or the Streamlit app
2. **Preprocess**: Run `preprocess.py` to normalize and augment
3. **Train**: Run `train.py` to update the model
4. **Test**: Use `inference.py` or the Live Demo tab

## Configuration

### Confidence Threshold
- Default: 70%
- Adjust via CLI flag `--threshold` or UI slider
- Higher = fewer false positives, may miss valid gestures

### TTS Settings
- Rate: 150 words/minute (adjustable in code)
- Cooldown: 2 seconds between announcements

## Troubleshooting

### Camera not working
```bash
# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### No hand detected
- Ensure good lighting
- Position hand clearly in frame
- Avoid cluttered backgrounds

### Low accuracy
- Collect more samples (50+ per gesture recommended)
- Use data augmentation (`--augment` flag)
- Train for more epochs

## License

MIT License - Feel free to use and modify.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Hand landmark detection
- [TensorFlow/Keras](https://www.tensorflow.org/) - Deep learning
- [Streamlit](https://streamlit.io/) - Web interface
- [pyttsx3](https://pyttsx3.readthedocs.io/) - Text-to-speech
