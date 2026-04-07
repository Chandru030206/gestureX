"""
realtime_inference.py
=====================
Real-time ASL alphabet recognition on Apple M1 using MPS.
Loads the trained asl_alphabet_m1.pth model, runs MediaPipe
hand detection, crops the hand, runs MobileNetV3 inference at 30+ FPS,
and renders a GestureX-style OpenCV overlay on the live webcam feed.

Usage:
    python3 realtime_inference.py
    python3 realtime_inference.py --model GestureX/models/ASL/asl_alphabet_m1_best.pth
    python3 realtime_inference.py --model path/to/model.pth --camera 0 --mirror

Controls:
    Q / ESC     — quit
    C           — clear current word buffer
    SPACE       — confirm current word into sentence
    S           — speak the full sentence (macOS say command)

Requirements:
    pip install opencv-python mediapipe torch torchvision
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import time
import json
import os
import argparse
import subprocess
from collections import deque
from pathlib import Path
from torchvision import transforms, models

# ─── CLI ARGS ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GestureX real-time ASL inference")
parser.add_argument("--model",  type=str,
    default="GestureX/models/ASL/asl_alphabet_m1_best.pth",
    help="Path to trained .pth weights file")
parser.add_argument("--meta",   type=str,
    default="GestureX/models/ASL/asl_alphabet_m1_meta.json",
    help="Path to metadata JSON (for class names)")
parser.add_argument("--camera", type=int, default=0, help="Camera device index")
parser.add_argument("--mirror", action="store_true", default=True,
    help="Horizontally flip frame (default True for selfie webcam)")
parser.add_argument("--conf_threshold", type=float, default=0.75,
    help="Minimum confidence to display a prediction (0–1)")
parser.add_argument("--stability_frames", type=int, default=5,
    help="Consecutive identical predictions before confirming a letter")
parser.add_argument("--letter_cooldown", type=float, default=0.9,
    help="Seconds to wait before adding same letter again")
args = parser.parse_args()

# ─── DEVICE ──────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ MPS (Apple Silicon GPU) active")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ CUDA active")
else:
    DEVICE = torch.device("cpu")
    print("⚠️  CPU mode — may be slow")

# ─── CLASS NAMES ─────────────────────────────────────────────
DEFAULT_CLASSES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

meta_path = Path(args.meta)
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    CLASS_NAMES = meta["classes"]
    print(f"✅ Loaded {len(CLASS_NAMES)} classes from metadata")
else:
    CLASS_NAMES = DEFAULT_CLASSES
    print(f"⚠️  No metadata file found — using default 29 ASL classes")
NUM_CLASSES = len(CLASS_NAMES)

# ─── MODEL ───────────────────────────────────────────────────
print(f"\n🧠 Loading model: {args.model}")
model = models.mobilenet_v3_large(weights=None)
in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.Hardswish(),
    nn.Dropout(p=0.3),
    nn.Linear(512, NUM_CLASSES),
)

model_path = Path(args.model)
if not model_path.exists():
    # Try the best checkpoint as fallback
    alt = model_path.parent / "asl_alphabet_m1_best.pth"
    if alt.exists():
        model_path = alt
        print(f"   ℹ️  Using best checkpoint: {model_path.name}")
    else:
        raise FileNotFoundError(
            f"Model not found at {args.model}\n"
            "Run train_asl_alphabet_m1.py first."
        )

ckpt = torch.load(str(model_path), map_location="cpu")
# Support both raw state_dict and checkpoint dict
if isinstance(ckpt, dict) and "model_state" in ckpt:
    state = ckpt["model_state"]
    print(f"   Checkpoint epoch: {ckpt.get('epoch','?')}  val_acc: {ckpt.get('val_acc','?'):.1f}%")
else:
    state = ckpt
model.load_state_dict(state)
model.to(DEVICE)
model.eval()
print(f"✅ Model ready on {DEVICE}")

# ─── TRANSFORMS ──────────────────────────────────────────────
# MUST match training transforms (no augmentation, just resize + normalize)
INFER_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ─── MEDIAPIPE HANDS ─────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,                # 0 = faster; 1 = more accurate
    min_detection_confidence=0.70,
    min_tracking_confidence=0.60,
)
print("✅ MediaPipe Hands ready")

# ─── INFERENCE HELPER ────────────────────────────────────────
@torch.inference_mode()
def run_inference(crop_bgr: np.ndarray):
    """Takes a BGR crop, returns (label_str, confidence_float 0-1)."""
    rgb   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = INFER_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    conf,  idx = torch.max(probs, dim=0)
    return CLASS_NAMES[idx.item()], conf.item()

# ─── STABILITY FILTER ─────────────────────────────────────────
# Only commit a letter when it appears STABILITY_FRAMES in a row
# and the confidence stays above threshold. Prevents flicker.
pred_history  = deque(maxlen=args.stability_frames)
stable_letter = None
last_committed_letter = ""
last_commit_time      = 0.0

def check_stability(letter: str, conf: float) -> str | None:
    """Returns the stable letter if confirmed, else None."""
    global stable_letter
    if conf < args.conf_threshold:
        pred_history.clear()
        return None
    pred_history.append(letter)
    if (len(pred_history) == args.stability_frames
            and len(set(pred_history)) == 1):
        stable_letter = letter
        return letter
    return None

# ─── WORD / SENTENCE STATE ────────────────────────────────────
fingerspell_buffer = []   # Live rolling letters
sentence_words     = []   # Confirmed words
last_letter_time   = 0.0

# Prefix-based word prediction (mirrors JS logic)
WORD_VOCAB = [
    "HELLO","HELP","HAPPY","HOME","HAVE","HEAR","HURRY",
    "PLEASE","PAIN","PATIENT","PLAY",
    "STOP","SORRY","SAFE","SEE","SCHOOL","SPEAK",
    "THANK","THINK","TIME","TODAY","TOMORROW",
    "GOOD","GO","GOODBYE",
    "YES","YOU","YOUR",
    "NO","NAME","NEED","NICE","NOW",
    "MORE","ME","MY","MOTHER",
    "WATER","WANT","WAIT","WORK","WHERE","WHEN","WHY","WHO","WHAT",
    "LOVE","LEARN","LIKE",
    "FAMILY","FATHER","FRIEND","FEEL","FOOD",
    "BAD","BYE","BETTER",
    "DANGER","DOCTOR","DRINK",
    "EMERGENCY","EAT","EASY",
    "KNOW","CALL",
]

def predict_word(prefix: str) -> str:
    if len(prefix) < 2:
        return prefix
    matches = [w for w in WORD_VOCAB if w.startswith(prefix.upper())]
    if matches:
        matches.sort(key=len)
        return matches[0]
    return prefix

# ─── COLORS (GestureX black & white palette) ─────────────────
BLACK  = (  0,   0,   0)
WHITE  = (255, 255, 255)
GRAY   = (180, 180, 180)
RED    = ( 50,  50, 200)   # BGR — for low confidence
GREEN  = ( 50, 200,  50)   # BGR — for high confidence
ACCENT = (220, 220, 220)

# ─── DRAW HELPERS ────────────────────────────────────────────
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX

def draw_panel(img, x, y, w, h, title: str, bg=WHITE, border=BLACK):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), border, 1)
    cv2.rectangle(img, (x, y), (x+w, y+20), border, -1)            # title bar
    cv2.putText(img, title, (x+6, y+14), FONT, 0.38, WHITE, 1)

def draw_conf_bar(img, x, y, w, conf: float, high_color=BLACK):
    bar_w = int(w * conf)
    cv2.rectangle(img, (x, y), (x+w, y+6), GRAY, -1)
    cv2.rectangle(img, (x, y), (x+bar_w, y+6), high_color, -1)

# ─── FPS TRACKER ─────────────────────────────────────────────
fps_buf    = deque(maxlen=30)
frame_time = time.perf_counter()

# ─── WEBCAM ──────────────────────────────────────────────────
cap = cv2.VideoCapture(args.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {args.camera}")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\n📷 Camera {args.camera} opened at {W}×{H}")
print("──────────────────────────────────────────────")
print("  Q / ESC  — Quit")
print("  C        — Clear word buffer")
print("  SPACE    — Confirm word into sentence")
print("  S        — Speak sentence")
print("──────────────────────────────────────────────\n")

# ─── MAIN LOOP ───────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    now       = time.perf_counter()
    fps_buf.append(1.0 / max(now - frame_time, 1e-6))
    frame_time = now
    fps        = sum(fps_buf) / len(fps_buf)

    if args.mirror:
        frame = cv2.flip(frame, 1)

    # ── MediaPipe hand detection ─────────────────────────────
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    predicted_letter = "—"
    confidence       = 0.0
    confirmed_letter = None
    hand_bbox        = None

    if results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks[0]

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame, lms, mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        # ── Compute bounding box ─────────────────────────────
        xs = [lm.x * W for lm in lms.landmark]
        ys = [lm.y * H for lm in lms.landmark]
        pad = 40
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(W, int(max(xs)) + pad)
        y2 = min(H, int(max(ys)) + pad)
        hand_bbox = (x1, y1, x2, y2)

        # ── Crop & infer ─────────────────────────────────────
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            predicted_letter, confidence = run_inference(crop)

            # Don't show 'nothing' as a positive prediction
            if predicted_letter in ("nothing",):
                predicted_letter = "—"
                confidence       = 0.0

        # ── Stability filter ─────────────────────────────────
        confirmed_letter = check_stability(predicted_letter, confidence)

        # ── Commit confirmed letter to buffer ─────────────────
        if (confirmed_letter
                and confirmed_letter not in ("—", "nothing", "del", "space")
                and (confirmed_letter != last_committed_letter
                     or now - last_letter_time > args.letter_cooldown)):
            fingerspell_buffer.append(confirmed_letter)
            last_committed_letter = confirmed_letter
            last_letter_time      = now

        elif confirmed_letter == "del" and fingerspell_buffer:
            fingerspell_buffer.pop()
            last_commit_time = now

        elif confirmed_letter == "space":
            # Finalise the current spelling as a word
            if fingerspell_buffer:
                word = predict_word("".join(fingerspell_buffer))
                sentence_words.append(word)
                fingerspell_buffer.clear()

        # Draw bounding box
        box_color = GREEN if confidence >= args.conf_threshold else RED
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"{confidence*100:.0f}%",
                    (x1, y1 - 8), FONT, 0.55, box_color, 1)

    else:
        # No hand: clear stability buffer
        pred_history.clear()

    # ─── OVERLAY PANELS (GestureX style) ─────────────────────
    PANEL_X  = W - 380
    PANEL_W  = 370
    PANEL_Y  = 20

    # ── Panel 1: DETECTED GESTURE ─────────────────────────────
    draw_panel(frame, PANEL_X, PANEL_Y, PANEL_W, 110,
               "DETECTED GESTURE")
    disp_letter = (confirmed_letter
                   if confirmed_letter and confirmed_letter not in ("nothing","—")
                   else predicted_letter)
    disp_color  = BLACK if confidence >= args.conf_threshold else GRAY
    cv2.putText(frame, disp_letter,
                (PANEL_X + 16, PANEL_Y + 80), FONT_BOLD,
                2.4, disp_color, 3)
    draw_conf_bar(frame, PANEL_X + 140, PANEL_Y + 65, 220, confidence)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (PANEL_X + 140, PANEL_Y + 90), FONT, 0.5, GRAY, 1)

    # ── Panel 2: FINGERSPELL (LIVE) ───────────────────────────
    PANEL_Y2 = PANEL_Y + 120
    draw_panel(frame, PANEL_X, PANEL_Y2, PANEL_W, 60,
               "FINGERSPELL (LIVE)")
    live_str = " ".join(fingerspell_buffer[-18:])   # show last 18 letters
    cv2.putText(frame, live_str,
                (PANEL_X + 8, PANEL_Y2 + 44), FONT_BOLD,
                0.75, BLACK, 2)

    # ── Panel 3: WORD PREDICTION ──────────────────────────────
    PANEL_Y3 = PANEL_Y2 + 70
    draw_panel(frame, PANEL_X, PANEL_Y3, PANEL_W, 50,
               "PREDICTED WORD")
    prefix = "".join(fingerspell_buffer)
    pred_w = predict_word(prefix) if prefix else "—"
    cv2.putText(frame, pred_w,
                (PANEL_X + 8, PANEL_Y3 + 36), FONT_BOLD,
                0.9, BLACK, 2)

    # ── Panel 4: CURRENT SENTENCE ─────────────────────────────
    PANEL_Y4 = PANEL_Y3 + 60
    draw_panel(frame, PANEL_X, PANEL_Y4, PANEL_W, 60,
               "CURRENT SENTENCE")
    sentence_str = " ".join(sentence_words)
    # Truncate long sentences for display
    if len(sentence_str) > 30:
        sentence_str = "…" + sentence_str[-29:]
    cv2.putText(frame, sentence_str,
                (PANEL_X + 8, PANEL_Y4 + 40), FONT,
                0.65, BLACK, 1)

    # ── Panel 5: CONTROLS ─────────────────────────────────────
    PANEL_Y5 = PANEL_Y4 + 70
    draw_panel(frame, PANEL_X, PANEL_Y5, PANEL_W, 80,
               "CONTROLS", bg=WHITE)
    for i, txt in enumerate(["SPACE: confirm word", "C: clear  DEL: backspace",
                              "S: speak  Q: quit"]):
        cv2.putText(frame, txt, (PANEL_X + 8, PANEL_Y5 + 28 + i * 18),
                    FONT, 0.42, GRAY, 1)

    # ── FPS & Device badge (top-left) ─────────────────────────
    badge = f"GestureX | {DEVICE.type.upper()} | {fps:.0f} FPS"
    (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.52, 1)
    cv2.rectangle(frame, (8, 8), (18 + bw, 28), BLACK, -1)
    cv2.putText(frame, badge, (12, 23), FONT, 0.52, WHITE, 1)

    # ── Stability counter dots ────────────────────────────────
    for i in range(args.stability_frames):
        filled = i < len(pred_history) and (
            len(pred_history) > i and pred_history[i] == predicted_letter)
        color  = BLACK if filled else GRAY
        cx = PANEL_X + 12 + i * 18
        cv2.circle(frame, (cx, PANEL_Y + 108), 5, color, -1)

    # ── Show frame ───────────────────────────────────────────
    cv2.imshow("GestureX — ASL Real-time Inference  (Q to quit)", frame)

    # ── Key handling ─────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):          # Q or ESC
        break
    elif key == ord('c'):              # Clear buffer
        fingerspell_buffer.clear()
        pred_history.clear()
    elif key == ord(' '):              # Confirm word
        if fingerspell_buffer:
            word = predict_word("".join(fingerspell_buffer))
            sentence_words.append(word)
            fingerspell_buffer.clear()
            print(f"✔ Word confirmed: {word}")
            print(f"  Sentence: {' '.join(sentence_words)}")
    elif key == ord('s'):              # Speak
        if sentence_words:
            text = " ".join(sentence_words)
            print(f"🔊 Speaking: {text}")
            subprocess.Popen(["say", text])

# ─── CLEANUP ─────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hands_detector.close()
print("\n✅ Session ended.")
if sentence_words:
    print(f"   Final sentence: {' '.join(sentence_words)}")
