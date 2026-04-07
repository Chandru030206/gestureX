"""
Speech to Gesture Engine
========================
Converts text/speech to sign language gestures.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SpeechToGestureEngine:
    """Converts text to sign language gestures with TTS."""
    
    def __init__(self, sign_db, audio_dir: str):
        self.sign_db = sign_db
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.tts_engine = self._init_tts()
    
    def _init_tts(self):
        """Initialize text-to-speech."""
        try:
            from gtts import gTTS
            return "gtts"
        except:
            return None
    
    def generate_audio(self, text: str) -> Optional[str]:
        """Generate audio file for text."""
        if not text or not self.tts_engine:
            return None
        
        try:
            cache_key = hashlib.md5(text.lower().encode()).hexdigest()[:12]
            filename = f"speech_{cache_key}.mp3"
            filepath = self.audio_dir / filename
            
            if filepath.exists():
                return filename
            
            from gtts import gTTS
            tts = gTTS(text=text, lang='en')
            tts.save(str(filepath))
            return filename
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    def text_to_gestures(self, text: str, language: str = "ASL") -> Dict[str, Any]:
        """Convert text to sign language gestures."""
        text = text.strip().lower()
        words = text.split()
        
        # Try to match the full phrase first
        full_match = self.sign_db.get_sign(language, text)
        if full_match:
            audio_file = self.generate_audio(text)
            return {
                "success": True,
                "type": "single",
                "original_text": text,
                "language": language,
                "gesture": full_match,
                "audio_url": f"/static/audio/{audio_file}" if audio_file else None
            }
        
        # Match individual words
        gestures = []
        unmatched = []
        
        for word in words:
            sign = self.sign_db.get_sign(language, word)
            if sign:
                # Remove emoji from sign data for professional output
                sign_copy = sign.copy()
                sign_copy["emoji"] = ""
                gestures.append(sign_copy)
            else:
                # Use fingerspelling for unmatched words
                fingerspelling = self.sign_db.get_fingerspelling(language, word)
                if fingerspelling:
                    gestures.append({
                        "word": word,
                        "type": "fingerspelling",
                        "letters": fingerspelling,
                        "description": f"Fingerspell: {word.upper()}",
                        "hand_shape": "Fingerspell each letter",
                        "emoji": ""
                    })
                    unmatched.append(word)
        
        if gestures:
            audio_file = self.generate_audio(text)
            return {
                "success": True,
                "type": "sentence",
                "original_text": text,
                "language": language,
                "word_count": len(words),
                "matched_count": len(words) - len(unmatched),
                "gestures": gestures,
                "unmatched_words": unmatched,
                "audio_url": f"/static/audio/{audio_file}" if audio_file else None
            }
        
        return {
            "success": False,
            "error": f"No signs found for: {text}",
            "suggestion": "Try simpler words or check spelling"
        }
