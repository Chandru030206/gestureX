# 🤟 Gesture ↔ Speech AI System

A bidirectional communication system that converts hand gestures to speech and speech/text to gesture visualizations.

## 🌟 Features

- **Gesture → Speech**: Real-time hand gesture recognition with text-to-speech output
- **Speech → Gesture**: Convert spoken/typed words to gesture visualizations
- **Pre-trained Model**: Uses Google's MediaPipe Gesture Recognizer (no training required!)
- **Clean UI**: Streamlit-based interface with option selection
- **7 Supported Gestures**: Closed Fist, Open Palm, Pointing Up, Thumb Down, Thumb Up, Victory, I Love You

## 📁 Project Structure

```
gesture-speech-ai/
├── app_new.py                 # Main Streamlit UI (use this!)
├── requirements.txt           # Python dependencies
├── gesture_recognizer.task    # Pre-trained model (download required)
├── backend/
│   ├── __init__.py
│   ├── load_models.py         # Model loading utilities
│   ├── inference.py           # Gesture recognition & TTS
│   └── speech_to_gesture.py   # Text/Speech to gesture mapping
├── assets/
│   ├── gesture_images/        # Generated gesture reference images
│   └── pretrained_model/      # Alternative model location
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Download Pre-trained Model

```bash
# Download MediaPipe Gesture Recognizer model
curl -O https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

### 3. Run the Application

```bash
# Start Streamlit app
streamlit run app_new.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### Option 1: Gesture → Speech
1. Select **"Gesture to Speech"** on the home screen
2. Click **"Start Camera"** to activate your webcam
3. Show hand gestures to the camera
4. The system will recognize the gesture and speak the meaning

### Option 2: Speech → Gesture
1. Select **"Speech to Gesture"** on the home screen
2. Type a word in the text box (e.g., "hello", "yes", "peace")
3. Click **"Find Gesture"** or use quick buttons
4. The system will show the corresponding gesture image

## 🖐️ Supported Gestures

| Gesture | Speech Output | Trigger Words |
|---------|---------------|---------------|
| ✊ Closed Fist | "Stop" | stop, wait, halt |
| ✋ Open Palm | "Hello" | hello, hi, bye, wave |
| ☝️ Pointing Up | "One moment" | one, wait, hold on |
| 👎 Thumb Down | "No" | no, bad, wrong, disagree |
| 👍 Thumb Up | "Yes" | yes, ok, good, great |
| ✌️ Victory | "Peace" | peace, victory, two |
| 🤟 I Love You | "I love you" | love, i love you |

## 🛠️ Backend API

### GestureRecognizer (backend/inference.py)

```python
from backend.inference import GestureRecognizer

# Initialize
recognizer = GestureRecognizer()
recognizer.initialize()

# Process frame
frame = cv2.imread("hand_image.jpg")
annotated_frame, result = recognizer.process_frame(frame)

print(f"Gesture: {result.gesture_name}")
print(f"Confidence: {result.confidence}")
print(f"Speech: {result.speech_text}")
```

### SpeechToGestureMapper (backend/speech_to_gesture.py)

```python
from backend.speech_to_gesture import SpeechToGestureMapper

# Initialize
mapper = SpeechToGestureMapper()

# Map text to gesture
result = mapper.map_text_to_gesture("hello")
print(f"Gesture: {result.gesture_name}")
print(f"Image path: {result.image_path}")
```

## 🔧 Configuration

### Confidence Threshold
Adjust in the UI or in code:
```python
recognizer.min_confidence = 0.7  # 70% confidence required
```

### Speech Cooldown
Prevent repeated speech:
```python
recognizer.speech_cooldown = 2.0  # 2 seconds between same gesture
```

## 🐛 Troubleshooting

### Camera not working
- Check camera permissions in System Preferences
- Try a different camera index: change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### Model not found
- Ensure `gesture_recognizer.task` is in the project root
- Or place it in `assets/pretrained_model/gesture_recognizer.task`

### TTS not working
- On macOS: TTS should work out of the box
- On Linux: Install `espeak`: `sudo apt-get install espeak`
- On Windows: Should work with default SAPI5

### Speech recognition not available
- Install: `pip install SpeechRecognition`
- Note: PyAudio may require additional system libraries

## 📦 Dependencies

- **tensorflow** >= 2.15.0 - Deep learning framework
- **mediapipe** >= 0.10.9 - Hand landmark & gesture detection
- **opencv-python** >= 4.8.0 - Computer vision
- **streamlit** >= 1.28.0 - Web UI framework
- **pyttsx3** >= 2.90 - Text-to-speech
- **numpy** >= 1.24.0 - Numerical computing
- **Pillow** >= 10.0.0 - Image processing

## 📄 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Google's ML pipeline framework
- [Streamlit](https://streamlit.io/) - Amazing web app framework
- Hand gesture recognition model by Google Research
