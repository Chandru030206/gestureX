"""
Fingerspelling API Module
=========================

API endpoints for fingerspelling-based name detection.

Endpoints:
- POST /api/fingerspell/start - Start a fingerspelling session
- POST /api/fingerspell/frame - Process a video frame
- POST /api/fingerspell/stop - Stop session and get result
- GET /api/fingerspell/state - Get current detection state
- POST /api/fingerspell/speak - Generate TTS for detected name
"""

import os
import base64
import hashlib
import logging
import numpy as np
import cv2
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# TTS
from gtts import gTTS

# Local imports
from fingerspelling_detector import (
    FingerspellingSession,
    FingerspellingResult,
    get_or_create_session,
    remove_session
)
from alphabet_classifier import LANGUAGE_ALPHABETS, get_classifier_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fingerspell", tags=["Fingerspelling"])

# Audio cache directory
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "assets", "audio", "fingerspell")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Try to import MediaPipe for hand detection
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available")

# Global hand detector
_hand_detector = None


def get_hand_detector():
    """Get or create MediaPipe hand detector"""
    global _hand_detector
    if _hand_detector is None and MEDIAPIPE_AVAILABLE:
        _hand_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    return _hand_detector


# =============================================================================
# Pydantic Models
# =============================================================================

class StartSessionRequest(BaseModel):
    language: str = "ASL"
    session_id: Optional[str] = "default"


class FrameRequest(BaseModel):
    image: str  # Base64 encoded image
    session_id: Optional[str] = "default"


class StopSessionRequest(BaseModel):
    session_id: Optional[str] = "default"


class SpeakRequest(BaseModel):
    name: str
    language: Optional[str] = "en"


class SessionResponse(BaseModel):
    session_id: str
    language: str
    status: str
    message: str


class FrameResponse(BaseModel):
    session_id: str
    state: str
    current_letter: Optional[str]
    current_confidence: Optional[float]
    detected_letters: List[str]
    partial_name: str
    is_complete: bool
    final_result: Optional[dict] = None


class FinalResultResponse(BaseModel):
    session_id: str
    detected_letters: List[str]
    detected_name: str
    confidence: float
    language: str
    audio_url: Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported sign languages for fingerspelling"""
    languages = []
    classifier_manager = get_classifier_manager()
    available_models = classifier_manager.list_available_models()
    
    for code, info in LANGUAGE_ALPHABETS.items():
        languages.append({
            "code": code,
            "name": info["name"],
            "alphabet_count": info["alphabet_count"],
            "has_model": code in available_models,
            "notes": info.get("notes", ""),
            "two_handed": info.get("two_handed", False)
        })
    
    return {"languages": languages}


@router.post("/start", response_model=SessionResponse)
async def start_session(request: StartSessionRequest):
    """
    Start a new fingerspelling detection session.
    Camera should be started on frontend after this call succeeds.
    """
    language = request.language.upper()
    session_id = request.session_id or "default"
    
    # Validate language
    if language not in LANGUAGE_ALPHABETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {language}. Supported: {list(LANGUAGE_ALPHABETS.keys())}"
        )
    
    # Create session
    session = get_or_create_session(session_id, language)
    session.set_language(language)
    session.start()
    
    logger.info(f"Started fingerspelling session {session_id} for {language}")
    
    return SessionResponse(
        session_id=session_id,
        language=language,
        status="started",
        message=f"Fingerspelling session started for {LANGUAGE_ALPHABETS[language]['name']}. Camera ready."
    )


@router.post("/frame", response_model=FrameResponse)
async def process_frame(request: FrameRequest):
    """
    Process a single video frame for alphabet detection.
    Expects base64 encoded image data.
    """
    session_id = request.session_id or "default"
    
    # Get session
    try:
        session = get_or_create_session(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Session error: {e}")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session not active. Call /start first.")
    
    # Decode image
    try:
        # Remove data URL prefix if present
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode error: {e}")
    
    # Detect hand landmarks
    landmarks = None
    
    if MEDIAPIPE_AVAILABLE:
        detector = get_hand_detector()
        if detector:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
    
    # Process landmarks
    state = session.process_landmarks(landmarks)
    
    # Build response
    is_complete = state.get("is_complete", False)
    final_result = state.get("final_result", None)
    
    last_pred = state.get("last_prediction", {})
    
    return FrameResponse(
        session_id=session_id,
        state=state.get("state", "unknown"),
        current_letter=last_pred.get("letter"),
        current_confidence=last_pred.get("confidence"),
        detected_letters=state.get("detected_letters", []),
        partial_name=state.get("partial_name", ""),
        is_complete=is_complete,
        final_result=final_result
    )


@router.post("/stop", response_model=FinalResultResponse)
async def stop_session(request: StopSessionRequest):
    """
    Stop the fingerspelling session and return final result.
    Also generates TTS audio for the detected name.
    """
    session_id = request.session_id or "default"
    
    try:
        session = get_or_create_session(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Session error: {e}")
    
    # Stop and get result
    result = session.stop()
    
    # Generate audio if name detected
    audio_url = None
    if result.detected_name:
        audio_url = await generate_name_audio(result.detected_name)
    
    logger.info(f"Session {session_id} stopped. Name: {result.detected_name}")
    
    return FinalResultResponse(
        session_id=session_id,
        detected_letters=result.detected_letters,
        detected_name=result.detected_name,
        confidence=result.confidence,
        language=result.language,
        audio_url=audio_url
    )


@router.get("/state/{session_id}")
async def get_session_state(session_id: str = "default"):
    """Get current state of a fingerspelling session"""
    try:
        session = get_or_create_session(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Session error: {e}")
    
    state = session.detector.get_current_state()
    state["is_active"] = session.is_active
    state["frame_count"] = session.frame_count
    
    return state


@router.delete("/session/{session_id}")
async def delete_session(session_id: str = "default"):
    """Delete a fingerspelling session"""
    remove_session(session_id)
    return {"status": "deleted", "session_id": session_id}


@router.post("/speak")
async def speak_name(request: SpeakRequest):
    """
    Generate TTS audio for a name.
    Returns URL to audio file.
    """
    if not request.name:
        raise HTTPException(status_code=400, detail="Name is required")
    
    audio_url = await generate_name_audio(request.name, request.language)
    
    return {
        "name": request.name,
        "message": f"Your name is {request.name}",
        "audio_url": audio_url
    }


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve generated audio file"""
    filepath = os.path.join(AUDIO_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(filepath, media_type="audio/mpeg")


# =============================================================================
# Helper Functions
# =============================================================================

async def generate_name_audio(name: str, lang: str = "en") -> str:
    """
    Generate TTS audio announcing the detected name.
    Returns URL to the audio file.
    """
    message = f"Your name is {name}"
    
    # Create hash for caching
    text_hash = hashlib.md5(f"{message}_{lang}".encode()).hexdigest()[:12]
    filename = f"name_{text_hash}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    
    # Generate if not cached
    if not os.path.exists(filepath):
        try:
            tts = gTTS(text=message, lang=lang, slow=False)
            tts.save(filepath)
            logger.info(f"Generated audio: {filename}")
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    return f"/api/fingerspell/audio/{filename}"
