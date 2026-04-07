"""
Multi-Language Sign Language Comparison API
Demonstrates how the SAME word is expressed differently across multiple sign languages
with visual gesture output and spoken audio explanation.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import hashlib
from gtts import gTTS

from sign_language_data import (
    SIGN_LANGUAGE_DATA,
    AVAILABLE_WORDS,
    SUPPORTED_LANGUAGES,
    get_sign_data,
    get_all_languages_for_word,
    get_language_info
)

router = APIRouter(prefix="/api/multilang", tags=["Multi-Language Comparison"])

# Audio cache directory
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "assets", "audio", "multilang")
os.makedirs(AUDIO_DIR, exist_ok=True)


class SignLanguageComparison(BaseModel):
    """Response model for single language sign data"""
    language: str
    language_name: str
    country: str
    word: str
    description: str
    gesture_steps: List[str]
    audio_text: str
    emoji: str
    hand_shape: str
    movement: str
    audio_url: Optional[str] = None


class MultiLanguageResponse(BaseModel):
    """Response model for multi-language comparison"""
    word: str
    total_languages: int
    comparisons: List[SignLanguageComparison]


class AvailableDataResponse(BaseModel):
    """Response for available words and languages"""
    languages: List[dict]
    words: List[str]


def generate_audio(text: str, language_code: str = "en") -> str:
    """
    Generate audio file for the given text using gTTS.
    Returns the filename of the generated audio.
    """
    # Create a unique hash for caching
    text_hash = hashlib.md5(f"{text}_{language_code}".encode()).hexdigest()[:12]
    filename = f"explanation_{text_hash}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    
    # Only generate if not cached
    if not os.path.exists(filepath):
        try:
            tts = gTTS(text=text, lang=language_code, slow=False)
            tts.save(filepath)
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None
    
    return filename


@router.get("/languages", response_model=List[dict])
async def get_supported_languages():
    """
    Get list of all supported sign languages with their details.
    """
    languages = []
    for lang_code in SUPPORTED_LANGUAGES:
        info = get_language_info(lang_code)
        if info:
            languages.append(info)
    return languages


@router.get("/words")
async def get_available_words():
    """
    Get list of all available words that can be compared.
    """
    return {"words": AVAILABLE_WORDS}


@router.get("/available")
async def get_available_data():
    """
    Get all available languages and words for the comparison system.
    """
    languages = []
    for lang_code in SUPPORTED_LANGUAGES:
        info = get_language_info(lang_code)
        if info:
            languages.append(info)
    
    return {
        "languages": languages,
        "words": AVAILABLE_WORDS
    }


@router.get("/compare/{word}")
async def compare_word_across_languages(
    word: str,
    languages: Optional[str] = None,
    generate_audio_files: bool = True
):
    """
    Compare how a word is signed across multiple (or all) sign languages.
    
    Args:
        word: The word to compare (e.g., "hello", "thank you")
        languages: Optional comma-separated list of language codes (e.g., "ASL,BSL,ISL")
                   If not provided, returns data for ALL languages
        generate_audio_files: Whether to generate audio explanation files
    
    Returns:
        Comparison data showing how the word is expressed in each language
    """
    word_lower = word.lower()
    
    if word_lower not in AVAILABLE_WORDS:
        raise HTTPException(
            status_code=404,
            detail=f"Word '{word}' not found. Available words: {AVAILABLE_WORDS}"
        )
    
    # Get all language data for this word
    all_data = get_all_languages_for_word(word_lower)
    
    # Filter by specific languages if requested
    if languages:
        lang_list = [l.strip().upper() for l in languages.split(",")]
        all_data = [d for d in all_data if d["language"] in lang_list]
    
    if not all_data:
        raise HTTPException(
            status_code=404,
            detail=f"No sign data found for word '{word}' in the specified languages"
        )
    
    # Generate audio for each language if requested
    comparisons = []
    for data in all_data:
        audio_url = None
        if generate_audio_files:
            audio_filename = generate_audio(data["audio_text"])
            if audio_filename:
                audio_url = f"/api/multilang/audio/{audio_filename}"
        
        comparisons.append({
            **data,
            "audio_url": audio_url
        })
    
    return {
        "word": word_lower,
        "total_languages": len(comparisons),
        "comparisons": comparisons
    }


@router.get("/sign/{language}/{word}")
async def get_single_sign(language: str, word: str, generate_audio_file: bool = True):
    """
    Get sign data for a specific language and word.
    
    Args:
        language: Language code (e.g., "ASL", "BSL", "ISL")
        word: The word to get (e.g., "hello")
        generate_audio_file: Whether to generate audio explanation
    
    Returns:
        Sign data including description, steps, and audio
    """
    data = get_sign_data(language, word)
    
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Sign for '{word}' not found in {language.upper()}"
        )
    
    audio_url = None
    if generate_audio_file:
        audio_filename = generate_audio(data["audio_text"])
        if audio_filename:
            audio_url = f"/api/multilang/audio/{audio_filename}"
    
    return {
        **data,
        "audio_url": audio_url
    }


@router.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Serve an audio explanation file.
    """
    filepath = os.path.join(AUDIO_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(filepath, media_type="audio/mpeg")


@router.get("/demo")
async def get_demo_comparison():
    """
    Get a demo comparison showing "HELLO" across all languages.
    This is a quick way to demonstrate the system's capabilities.
    """
    return await compare_word_across_languages("hello", generate_audio_files=True)
