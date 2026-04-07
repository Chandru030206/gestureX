"""
GestureX - Multi-Language Sign Language Communication Platform
==============================================================

Main FastAPI application supporting 300+ sign languages worldwide.

Features:
- Gesture → Speech: Webcam gesture recognition to spoken text
- Speech → Gesture: Text/speech to sign language gestures
- Multi-language support: ASL, BSL, ISL, JSL, Auslan, LSF, and more
- Fingerspelling fallback for unknown words

API Routes:
- POST /predict_gesture - Gesture recognition from webcam
- POST /speech_to_gesture - Convert text to sign gestures  
- POST /set_language - Set active sign language
- GET /sign/{lang}/{word} - Get sign resource for a word
- GET /languages - List all supported languages
- GET /health - Health check
"""

import os
import io
import base64
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Local imports
from inference_gesture import GestureRecognizer
from speech_to_gesture import SpeechToGestureEngine
from sign_language_db import SignLanguageDatabase, SUPPORTED_LANGUAGES
from multilang_api import router as multilang_router
from fingerspelling_api import router as fingerspelling_router

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
AUDIO_DIR = ASSETS_DIR / "audio"
IMAGES_DIR = ASSETS_DIR / "gesture_images"
VIDEOS_DIR = ASSETS_DIR / "gesture_videos"
DICTIONARIES_DIR = BASE_DIR / "sign_dictionaries"
MODELS_DIR = BASE_DIR / "pretrained_models"

# Create directories
for d in [AUDIO_DIR, IMAGES_DIR, VIDEOS_DIR, DICTIONARIES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="GestureX API",
    description="Multi-Language Sign Language Communication Platform",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")
app.mount("/static/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

# Include Multi-Language Comparison Router
app.include_router(multilang_router)

# Include Fingerspelling Router
app.include_router(fingerspelling_router)

# =============================================================================
# Global State & Services
# =============================================================================

gesture_recognizer: Optional[GestureRecognizer] = None
gesture_recognizers: Dict[str, GestureRecognizer] = {}
speech_to_gesture_engine: Optional[SpeechToGestureEngine] = None
sign_db: Optional[SignLanguageDatabase] = None

# Session state
session_state = {
    "current_language": "ASL",
    "mode": None,
    "session_start": datetime.now().isoformat()
}

# =============================================================================
# Pydantic Models
# =============================================================================

class FrameData(BaseModel):
    image: str  # Base64 encoded

class TextInput(BaseModel):
    text: str
    language: Optional[str] = "ASL"

class LanguageSelect(BaseModel):
    language: str

class ModeSelect(BaseModel):
    mode: str  # 'gesture_to_speech' or 'speech_to_gesture'

# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup():
    global gesture_recognizer, gesture_recognizers, speech_to_gesture_engine, sign_db
    
    logger.info("🚀 Starting GestureX Backend...")
    
    # Initialize sign language database
    sign_db = SignLanguageDatabase(str(DICTIONARIES_DIR))
    logger.info(f"✓ Sign language database loaded ({len(SUPPORTED_LANGUAGES)} languages)")
    
    # Initialize gesture recognizers (language-specific)
    # NOTE: At the moment we only have one MediaPipe model file in the repo.
    # This scaffolds *independent per-language recognizers* so different model
    # files can be added later without mixing languages.
    try:
        for lang in SUPPORTED_LANGUAGES.keys():
            try:
                gesture_recognizers[lang] = GestureRecognizer(language=lang)
            except Exception as e:
                logger.warning(f"Gesture recognizer init failed for {lang}: {e}")

        # Backward-compatible default
        gesture_recognizer = gesture_recognizers.get("ASL") or next(iter(gesture_recognizers.values()), None)
        logger.info(f"✓ Gesture recognizers initialized ({len(gesture_recognizers)} languages)")
    except Exception as e:
        logger.error(f"✗ Gesture recognizers failed: {e}")
    
    # Initialize speech-to-gesture engine
    try:
        speech_to_gesture_engine = SpeechToGestureEngine(sign_db, str(AUDIO_DIR))
        logger.info("✓ Speech-to-gesture engine initialized")
    except Exception as e:
        logger.error(f"✗ Speech-to-gesture engine failed: {e}")
    
    logger.info("✅ GestureX Backend ready!")

# =============================================================================
# API Routes
# =============================================================================

@app.get("/")
async def root():
    return {
        "name": "GestureX API",
        "version": "2.0.0",
        "description": "Multi-Language Sign Language Communication Platform",
        "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
        "endpoints": {
            "predict_gesture": "POST /predict_gesture",
            "speech_to_gesture": "POST /speech_to_gesture",
            "set_language": "POST /set_language",
            "set_mode": "POST /set_mode",
            "get_sign": "GET /sign/{lang}/{word}",
            "languages": "GET /languages",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "services": {
            "gesture_recognizer": gesture_recognizer is not None,
            "speech_to_gesture": speech_to_gesture_engine is not None,
            "sign_database": sign_db is not None
        },
        "session": session_state,
        "languages_loaded": len(SUPPORTED_LANGUAGES)
    }

@app.get("/languages")
async def get_languages():
    """Get all supported sign languages with details."""
    return {
        "languages": SUPPORTED_LANGUAGES,
        "current": session_state["current_language"],
        "total": len(SUPPORTED_LANGUAGES)
    }

@app.post("/set_language")
async def set_language(data: LanguageSelect):
    """Set the active sign language."""
    lang = data.language.upper()
    
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {lang}. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    
    session_state["current_language"] = lang
    # Keep default recognizer pointer aligned with current language for any code
    # paths still using `gesture_recognizer` directly.
    global gesture_recognizer
    try:
        if "gesture_recognizers" in globals() and isinstance(gesture_recognizers, dict):
            gesture_recognizer = gesture_recognizers.get(lang) or gesture_recognizer
    except Exception:
        pass
    logger.info(f"Language set to: {lang}")
    
    return {
        "success": True,
        "language": lang,
        "language_info": SUPPORTED_LANGUAGES[lang]
    }

@app.post("/set_mode")
async def set_mode(data: ModeSelect):
    """Set the operation mode."""
    valid_modes = ['gesture_to_speech', 'speech_to_gesture']
    
    if data.mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Use: {valid_modes}")
    
    session_state["mode"] = data.mode
    return {"success": True, "mode": data.mode}

@app.post("/predict_gesture")
async def predict_gesture(frame_data: FrameData):
    """
    Recognize gesture from webcam frame.
    Returns predicted sign, text, and audio.
    """
    if not gesture_recognizers and gesture_recognizer is None:
        raise HTTPException(status_code=500, detail="Gesture recognizer not initialized")
    
    try:
        # Decode base64 image
        image_data = frame_data.image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "error": "Failed to decode image"}
        
        # Recognize gesture using the currently selected language model
        selected_lang = session_state.get("current_language", "ASL")
        recognizer = gesture_recognizers.get(selected_lang) or gesture_recognizer
        if recognizer is None:
            raise HTTPException(status_code=500, detail="Gesture recognizer not initialized")

        result = recognizer.recognize(frame)
        
        if result["success"]:
            # Generate audio for the recognized text
            audio_url = None
            if speech_to_gesture_engine and result.get("text"):
                audio_file = speech_to_gesture_engine.generate_audio(result["text"])
                if audio_file:
                    audio_url = f"/static/audio/{audio_file}"
            
            return {
                "success": True,
                "gesture": result["gesture"],
                "confidence": result["confidence"],
                "text": result["text"],
                "type": result.get("type", "word"),
                "audio_url": audio_url,
                "language": session_state["current_language"]
            }
        else:
            return {"success": False, "error": result.get("error", "No gesture detected")}
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/speech_to_gesture")
async def speech_to_gesture(data: TextInput):
    """
    Convert text to sign language gesture.
    Returns gesture images/videos/instructions.
    """
    if speech_to_gesture_engine is None:
        raise HTTPException(status_code=500, detail="Speech-to-gesture engine not initialized")
    
    try:
        text = data.text.strip()
        language = data.language.upper() if data.language else session_state["current_language"]
        
        if not text:
            return {"success": False, "error": "Empty text"}
        
        # Get sign gestures for the text
        result = speech_to_gesture_engine.text_to_gestures(text, language)
        
        return result
        
    except Exception as e:
        logger.error(f"Speech-to-gesture error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/sign/{lang}/{word}")
async def get_sign_resource(lang: str, word: str):
    """Get sign resource (image/video/description) for a specific word."""
    if sign_db is None:
        raise HTTPException(status_code=500, detail="Sign database not initialized")
    
    lang = lang.upper()
    word = word.lower()
    
    result = sign_db.get_sign(lang, word)
    
    if result:
        return result
    else:
        # Return fingerspelling fallback
        fingerspelling = sign_db.get_fingerspelling(lang, word)
        return {
            "success": True,
            "word": word,
            "language": lang,
            "type": "fingerspelling",
            "letters": fingerspelling,
            "message": f"Word '{word}' not found. Showing fingerspelling."
        }

@app.get("/vocabulary/{lang}")
async def get_vocabulary(lang: str):
    """Get all available words for a language."""
    if sign_db is None:
        raise HTTPException(status_code=500, detail="Sign database not initialized")
    
    lang = lang.upper()
    vocab = sign_db.get_vocabulary(lang)
    
    return {
        "language": lang,
        "total_words": len(vocab),
        "categories": vocab
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
