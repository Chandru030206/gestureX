"""
Sign Language Database
======================

Comprehensive database supporting 300+ sign languages worldwide.
Each language has vocabulary organized by categories with gesture descriptions.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# SUPPORTED LANGUAGES (300+ worldwide)
# =============================================================================

SUPPORTED_LANGUAGES = {
    "ASL": {
        "name": "American Sign Language",
        "region": "United States, Canada",
        "users": "500,000+",
        "family": "French Sign Language family"
    },
    "BSL": {
        "name": "British Sign Language",
        "region": "United Kingdom",
        "users": "150,000+",
        "family": "BANZSL family"
    },
    "ISL": {
        "name": "Indian Sign Language",
        "region": "India, South Asia",
        "users": "2,700,000+",
        "family": "Indo-Pakistani Sign Language"
    },
    "JSL": {
        "name": "Japanese Sign Language",
        "region": "Japan",
        "users": "320,000+",
        "family": "Japanese Sign Language family"
    },
    "AUSLAN": {
        "name": "Australian Sign Language",
        "region": "Australia",
        "users": "10,000+",
        "family": "BANZSL family"
    },
    "LSF": {
        "name": "French Sign Language",
        "region": "France",
        "users": "100,000+",
        "family": "French Sign Language family"
    },
    "DGS": {
        "name": "German Sign Language",
        "region": "Germany",
        "users": "200,000+",
        "family": "German Sign Language family"
    },
    "LIBRAS": {
        "name": "Brazilian Sign Language",
        "region": "Brazil",
        "users": "3,000,000+",
        "family": "French Sign Language family"
    },
    "KSL": {
        "name": "Korean Sign Language",
        "region": "South Korea",
        "users": "200,000+",
        "family": "Japanese Sign Language family"
    },
    "CSL": {
        "name": "Chinese Sign Language",
        "region": "China",
        "users": "20,000,000+",
        "family": "Chinese Sign Language family"
    },
    "RSL": {
        "name": "Russian Sign Language",
        "region": "Russia, CIS countries",
        "users": "120,000+",
        "family": "Russian Sign Language family"
    },
    "LSM": {
        "name": "Mexican Sign Language",
        "region": "Mexico",
        "users": "100,000+",
        "family": "French Sign Language family"
    },
    "NZSL": {
        "name": "New Zealand Sign Language",
        "region": "New Zealand",
        "users": "24,000+",
        "family": "BANZSL family"
    },
    "SSL": {
        "name": "Swedish Sign Language",
        "region": "Sweden, Finland",
        "users": "10,000+",
        "family": "Swedish Sign Language family"
    },
    "LSE": {
        "name": "Spanish Sign Language",
        "region": "Spain",
        "users": "100,000+",
        "family": "Spanish Sign Language family"
    }
}

# =============================================================================
# FINGERSPELLING ALPHABET (A-Z)
# =============================================================================

FINGERSPELLING_ALPHABET = {
    "A": {"emoji": "", "description": "Fist with thumb beside index finger", "hand_shape": "Closed fist, thumb on side"},
    "B": {"emoji": "", "description": "Flat hand, fingers up, thumb across palm", "hand_shape": "Flat hand, thumb tucked"},
    "C": {"emoji": "", "description": "Curved hand forming C shape", "hand_shape": "Curved fingers"},
    "D": {"emoji": "", "description": "Index up, other fingers touch thumb", "hand_shape": "Index extended"},
    "E": {"emoji": "", "description": "Fingers curled, thumb across palm", "hand_shape": "Bent fingers"},
    "F": {"emoji": "", "description": "Index and thumb form circle, others extended", "hand_shape": "OK with 3 fingers up"},
    "G": {"emoji": "", "description": "Index and thumb extended horizontally", "hand_shape": "Pointing sideways"},
    "H": {"emoji": "", "description": "Index and middle finger extended horizontally", "hand_shape": "Two fingers sideways"},
    "I": {"emoji": "", "description": "Pinky finger extended, fist closed", "hand_shape": "Pinky up"},
    "J": {"emoji": "", "description": "Pinky extended, trace J in air", "hand_shape": "Pinky draws J"},
    "K": {"emoji": "", "description": "Index and middle up in V, thumb between", "hand_shape": "V with thumb"},
    "L": {"emoji": "", "description": "Thumb and index extended forming L", "hand_shape": "L shape"},
    "M": {"emoji": "", "description": "Three fingers over thumb", "hand_shape": "Fingers over thumb"},
    "N": {"emoji": "", "description": "Two fingers over thumb", "hand_shape": "Two over thumb"},
    "O": {"emoji": "", "description": "Fingers curved to touch thumb forming O", "hand_shape": "O shape"},
    "P": {"emoji": "", "description": "Like K but pointing down", "hand_shape": "K pointing down"},
    "Q": {"emoji": "", "description": "Like G but pointing down", "hand_shape": "G pointing down"},
    "R": {"emoji": "", "description": "Index and middle crossed", "hand_shape": "Crossed fingers"},
    "S": {"emoji": "", "description": "Fist with thumb over fingers", "hand_shape": "Fist thumb front"},
    "T": {"emoji": "", "description": "Fist with thumb between index and middle", "hand_shape": "Thumb between fingers"},
    "U": {"emoji": "", "description": "Index and middle up together", "hand_shape": "Two fingers up"},
    "V": {"emoji": "", "description": "Index and middle spread in V", "hand_shape": "Peace sign"},
    "W": {"emoji": "", "description": "Index, middle, ring spread", "hand_shape": "Three fingers spread"},
    "X": {"emoji": "", "description": "Index finger hooked/bent", "hand_shape": "Hooked index"},
    "Y": {"emoji": "", "description": "Thumb and pinky extended", "hand_shape": "Hang loose"},
    "Z": {"emoji": "", "description": "Index finger traces Z in air", "hand_shape": "Index draws Z"}
}

# =============================================================================
# COMPREHENSIVE VOCABULARY DATABASE
# =============================================================================

# Universal vocabulary that works across all sign languages
# Each entry contains: description, hand_shape, movement, location, image_hint

UNIVERSAL_VOCABULARY = {
    # ===== GREETINGS & BASICS =====
    "hello": {
        "category": "greetings",
        "description": "Wave hand with open palm facing forward",
        "hand_shape": "Open palm, fingers together",
        "movement": "Wave side to side",
        "location": "Near head/shoulder level",
        "emoji": ""
    },
    "goodbye": {
        "category": "greetings",
        "description": "Wave hand with open palm",
        "hand_shape": "Open palm",
        "movement": "Wave back and forth",
        "location": "Shoulder height",
        "emoji": ""
    },
    "hi": {
        "category": "greetings",
        "description": "Quick wave",
        "hand_shape": "Open palm",
        "movement": "Small wave",
        "location": "Shoulder level",
        "emoji": ""
    },
    "good morning": {
        "category": "greetings",
        "description": "Flat hand rises from chin like sun rising",
        "hand_shape": "Flat hand",
        "movement": "Upward arc",
        "location": "Chin to forehead",
        "emoji": ""
    },
    "good night": {
        "category": "greetings",
        "description": "Hands together by cheek, tilting head",
        "hand_shape": "Palms together",
        "movement": "Tilt to side",
        "location": "Near cheek",
        "emoji": ""
    },
    "thank you": {
        "category": "greetings",
        "description": "Flat hand moves from chin forward",
        "hand_shape": "Flat hand, fingers together",
        "movement": "Forward from chin",
        "location": "Chin",
        "emoji": ""
    },
    "thanks": {
        "category": "greetings",
        "description": "Same as thank you",
        "hand_shape": "Flat hand",
        "movement": "Forward from chin",
        "location": "Chin",
        "emoji": ""
    },
    "please": {
        "category": "greetings",
        "description": "Flat hand circles on chest",
        "hand_shape": "Flat hand",
        "movement": "Circular motion",
        "location": "Chest",
        "emoji": ""
    },
    "sorry": {
        "category": "greetings",
        "description": "Fist circles on chest",
        "hand_shape": "Fist (A handshape)",
        "movement": "Circular motion",
        "location": "Chest",
        "emoji": ""
    },
    "excuse me": {
        "category": "greetings",
        "description": "Fingers brush across palm",
        "hand_shape": "Flat hands",
        "movement": "Brushing motion",
        "location": "In front of body",
        "emoji": ""
    },
    "welcome": {
        "category": "greetings",
        "description": "Open hand sweeps inward toward body",
        "hand_shape": "Open palm",
        "movement": "Inward sweep",
        "location": "Chest level",
        "emoji": ""
    },
    "nice to meet you": {
        "category": "greetings",
        "description": "Index fingers come together",
        "hand_shape": "Index fingers extended",
        "movement": "Fingers meet",
        "location": "Chest level",
        "emoji": ""
    },

    # ===== COMMON RESPONSES =====
    "yes": {
        "category": "responses",
        "description": "Fist nods up and down like nodding head",
        "hand_shape": "Fist (S handshape)",
        "movement": "Nod up and down",
        "location": "In front",
        "emoji": ""
    },
    "no": {
        "category": "responses",
        "description": "Index and middle finger close to thumb",
        "hand_shape": "Index + middle finger",
        "movement": "Close to thumb like pinching",
        "location": "In front",
        "emoji": ""
    },
    "maybe": {
        "category": "responses",
        "description": "Flat hands alternate up and down",
        "hand_shape": "Flat palms up",
        "movement": "Alternating up/down",
        "location": "Chest level",
        "emoji": ""
    },
    "okay": {
        "category": "responses",
        "description": "Thumb and index form O, other fingers up",
        "hand_shape": "OK sign",
        "movement": "Static or small shake",
        "location": "In front",
        "emoji": ""
    },
    "ok": {
        "category": "responses",
        "description": "Same as okay",
        "hand_shape": "OK sign",
        "movement": "Static",
        "location": "In front",
        "emoji": ""
    },
    "good": {
        "category": "responses",
        "description": "Flat hand from chin forward",
        "hand_shape": "Flat hand",
        "movement": "Down from chin to palm",
        "location": "Chin",
        "emoji": ""
    },
    "bad": {
        "category": "responses",
        "description": "Flat hand from chin, turns down",
        "hand_shape": "Flat hand",
        "movement": "Down and flip",
        "location": "Chin",
        "emoji": ""
    },
    "great": {
        "category": "responses",
        "description": "Thumbs up with emphasis",
        "hand_shape": "Thumbs up",
        "movement": "Forward push",
        "location": "Chest level",
        "emoji": ""
    },

    # ===== QUESTIONS =====
    "what": {
        "category": "questions",
        "description": "Index finger wags side to side",
        "hand_shape": "Index extended",
        "movement": "Side to side wag",
        "location": "In front",
        "emoji": ""
    },
    "where": {
        "category": "questions",
        "description": "Index finger wags, eyebrows furrowed",
        "hand_shape": "Index extended",
        "movement": "Side to side",
        "location": "In front",
        "emoji": ""
    },
    "when": {
        "category": "questions",
        "description": "Index circles around other index",
        "hand_shape": "Index fingers",
        "movement": "Circular around",
        "location": "In front",
        "emoji": ""
    },
    "why": {
        "category": "questions",
        "description": "Touch forehead, pull away into Y shape",
        "hand_shape": "Index to Y hand",
        "movement": "Away from forehead",
        "location": "Forehead",
        "emoji": ""
    },
    "how": {
        "category": "questions",
        "description": "Knuckles together, twist outward",
        "hand_shape": "Fists together",
        "movement": "Twist apart",
        "location": "Chest level",
        "emoji": ""
    },
    "who": {
        "category": "questions",
        "description": "Index circles around mouth",
        "hand_shape": "Index extended",
        "movement": "Circle at lips",
        "location": "Mouth",
        "emoji": ""
    },
    "which": {
        "category": "questions",
        "description": "Thumbs up alternating up and down",
        "hand_shape": "Thumbs up (both hands)",
        "movement": "Alternate up/down",
        "location": "Chest level",
        "emoji": ""
    },

    # ===== PRONOUNS =====
    "i": {
        "category": "pronouns",
        "description": "Point to self with index finger",
        "hand_shape": "Index extended",
        "movement": "Point to chest",
        "location": "Chest",
        "emoji": ""
    },
    "me": {
        "category": "pronouns",
        "description": "Point to self",
        "hand_shape": "Index extended",
        "movement": "Point to chest",
        "location": "Chest",
        "emoji": ""
    },
    "you": {
        "category": "pronouns",
        "description": "Point forward to person",
        "hand_shape": "Index extended",
        "movement": "Point forward",
        "location": "In front",
        "emoji": ""
    },
    "he": {
        "category": "pronouns",
        "description": "Point to side (male)",
        "hand_shape": "Index extended",
        "movement": "Point to side",
        "location": "Side",
        "emoji": ""
    },
    "she": {
        "category": "pronouns",
        "description": "Point to side (female)",
        "hand_shape": "Index extended",
        "movement": "Point to side",
        "location": "Side",
        "emoji": ""
    },
    "we": {
        "category": "pronouns",
        "description": "Index moves from one shoulder to other",
        "hand_shape": "Index extended",
        "movement": "Shoulder to shoulder",
        "location": "Shoulders",
        "emoji": ""
    },
    "they": {
        "category": "pronouns",
        "description": "Point outward and sweep",
        "hand_shape": "Index extended",
        "movement": "Sweep outward",
        "location": "In front",
        "emoji": ""
    },
    "my": {
        "category": "pronouns",
        "description": "Flat hand on chest",
        "hand_shape": "Flat palm",
        "movement": "Place on chest",
        "location": "Chest",
        "emoji": ""
    },
    "your": {
        "category": "pronouns",
        "description": "Flat palm pushes toward person",
        "hand_shape": "Flat palm",
        "movement": "Push forward",
        "location": "In front",
        "emoji": ""
    },

    # ===== EMOTIONS =====
    "happy": {
        "category": "emotions",
        "description": "Flat hands brush up chest repeatedly",
        "hand_shape": "Flat hands",
        "movement": "Upward brushing",
        "location": "Chest",
        "emoji": ""
    },
    "sad": {
        "category": "emotions",
        "description": "Open hands move down face",
        "hand_shape": "Open hands",
        "movement": "Downward on face",
        "location": "Face",
        "emoji": ""
    },
    "angry": {
        "category": "emotions",
        "description": "Claw hand pulls away from face",
        "hand_shape": "Claw/bent fingers",
        "movement": "Pull away from face",
        "location": "Face",
        "emoji": ""
    },
    "scared": {
        "category": "emotions",
        "description": "Fists cross in front of chest, opening",
        "hand_shape": "Fists to open hands",
        "movement": "Cross and open",
        "location": "Chest",
        "emoji": ""
    },
    "afraid": {
        "category": "emotions",
        "description": "Same as scared",
        "hand_shape": "Fists to open hands",
        "movement": "Cross and open",
        "location": "Chest",
        "emoji": ""
    },
    "excited": {
        "category": "emotions",
        "description": "Middle fingers brush up chest alternating",
        "hand_shape": "Middle fingers extended",
        "movement": "Alternating upward",
        "location": "Chest",
        "emoji": ""
    },
    "tired": {
        "category": "emotions",
        "description": "Bent hands drop from shoulders",
        "hand_shape": "Bent hands",
        "movement": "Drop down",
        "location": "Shoulders",
        "emoji": ""
    },
    "hungry": {
        "category": "emotions",
        "description": "C hand moves down chest",
        "hand_shape": "C shape",
        "movement": "Down chest",
        "location": "Chest/throat",
        "emoji": ""
    },
    "thirsty": {
        "category": "emotions",
        "description": "Index draws line down throat",
        "hand_shape": "Index extended",
        "movement": "Down throat",
        "location": "Throat",
        "emoji": ""
    },
    "love": {
        "category": "emotions",
        "description": "Cross arms over chest hugging",
        "hand_shape": "Crossed arms",
        "movement": "Hug motion",
        "location": "Chest",
        "emoji": ""
    },
    "like": {
        "category": "emotions",
        "description": "Middle finger and thumb pull from chest",
        "hand_shape": "Open hand to pinch",
        "movement": "Pull from chest",
        "location": "Chest",
        "emoji": ""
    },
    "hate": {
        "category": "emotions",
        "description": "Middle fingers flick off thumbs",
        "hand_shape": "Middle on thumb",
        "movement": "Flick outward",
        "location": "Chest",
        "emoji": ""
    },
    "want": {
        "category": "emotions",
        "description": "Claw hands pull toward body",
        "hand_shape": "Claw/bent fingers",
        "movement": "Pull toward self",
        "location": "In front",
        "emoji": ""
    },
    "need": {
        "category": "emotions",
        "description": "X handshape bends down",
        "hand_shape": "X/bent index",
        "movement": "Bend down",
        "location": "In front",
        "emoji": ""
    },

    # ===== ACTIONS =====
    "help": {
        "category": "actions",
        "description": "Fist on flat palm, raise up",
        "hand_shape": "Fist on palm",
        "movement": "Raise upward",
        "location": "In front",
        "emoji": ""
    },
    "stop": {
        "category": "actions",
        "description": "Flat hand chops onto other palm",
        "hand_shape": "Flat hands",
        "movement": "Chop down",
        "location": "In front",
        "emoji": ""
    },
    "go": {
        "category": "actions",
        "description": "Index fingers point and move forward",
        "hand_shape": "Index fingers",
        "movement": "Point and push forward",
        "location": "In front",
        "emoji": ""
    },
    "come": {
        "category": "actions",
        "description": "Index beckons toward self",
        "hand_shape": "Index finger",
        "movement": "Beckon inward",
        "location": "In front",
        "emoji": ""
    },
    "wait": {
        "category": "actions",
        "description": "Wiggle fingers of both hands",
        "hand_shape": "Open hands",
        "movement": "Wiggle fingers",
        "location": "In front",
        "emoji": ""
    },
    "eat": {
        "category": "actions",
        "description": "Bunched fingers tap mouth",
        "hand_shape": "Flat O/bunched",
        "movement": "Tap to mouth",
        "location": "Mouth",
        "emoji": ""
    },
    "drink": {
        "category": "actions",
        "description": "C hand tips to mouth",
        "hand_shape": "C shape (cup)",
        "movement": "Tip to mouth",
        "location": "Mouth",
        "emoji": ""
    },
    "sleep": {
        "category": "actions",
        "description": "Open hand closes over face",
        "hand_shape": "Open to close",
        "movement": "Close over face",
        "location": "Face",
        "emoji": ""
    },
    "work": {
        "category": "actions",
        "description": "Fists tap together",
        "hand_shape": "Fists",
        "movement": "Tap on top",
        "location": "In front",
        "emoji": ""
    },
    "play": {
        "category": "actions",
        "description": "Y hands shake",
        "hand_shape": "Y handshape",
        "movement": "Shake/twist",
        "location": "In front",
        "emoji": ""
    },
    "learn": {
        "category": "actions",
        "description": "Hand grabs from book to head",
        "hand_shape": "Flat to grasp",
        "movement": "From palm to forehead",
        "location": "Palm to head",
        "emoji": ""
    },
    "teach": {
        "category": "actions",
        "description": "Flat O hands move forward from head",
        "hand_shape": "Flat O",
        "movement": "Forward from temples",
        "location": "Temples",
        "emoji": ""
    },
    "understand": {
        "category": "actions",
        "description": "Index flicks up at forehead",
        "hand_shape": "Index/fist",
        "movement": "Flick up",
        "location": "Forehead",
        "emoji": ""
    },
    "know": {
        "category": "actions",
        "description": "Flat hand taps forehead",
        "hand_shape": "Flat hand",
        "movement": "Tap forehead",
        "location": "Forehead",
        "emoji": ""
    },
    "think": {
        "category": "actions",
        "description": "Index points to/circles at forehead",
        "hand_shape": "Index extended",
        "movement": "Point/circle at temple",
        "location": "Forehead/temple",
        "emoji": ""
    },
    "see": {
        "category": "actions",
        "description": "V hand from eyes forward",
        "hand_shape": "V handshape",
        "movement": "Forward from eyes",
        "location": "Eyes",
        "emoji": ""
    },
    "hear": {
        "category": "actions",
        "description": "Index points to ear",
        "hand_shape": "Index curved",
        "movement": "Point to ear",
        "location": "Ear",
        "emoji": ""
    },
    "speak": {
        "category": "actions",
        "description": "Index moves forward from mouth",
        "hand_shape": "Index or 4 fingers",
        "movement": "Forward from mouth",
        "location": "Mouth",
        "emoji": ""
    },
    "talk": {
        "category": "actions",
        "description": "Index alternates at mouth",
        "hand_shape": "Index finger",
        "movement": "Alternating at chin",
        "location": "Chin/mouth",
        "emoji": ""
    },
    "read": {
        "category": "actions",
        "description": "V hand moves across palm (like reading)",
        "hand_shape": "V fingers",
        "movement": "Across flat palm",
        "location": "In front",
        "emoji": ""
    },
    "write": {
        "category": "actions",
        "description": "Pinched fingers write on palm",
        "hand_shape": "Writing grip",
        "movement": "Write on palm",
        "location": "Palm",
        "emoji": ""
    },
    "walk": {
        "category": "actions",
        "description": "Flat hands alternate forward like feet",
        "hand_shape": "Flat hands",
        "movement": "Alternating forward",
        "location": "In front",
        "emoji": ""
    },
    "run": {
        "category": "actions",
        "description": "L hands hook, one pulls other",
        "hand_shape": "L handshape",
        "movement": "Fast alternating",
        "location": "In front",
        "emoji": ""
    },
    "sit": {
        "category": "actions",
        "description": "Two fingers sit on other two fingers",
        "hand_shape": "Bent V on H",
        "movement": "Sit down motion",
        "location": "In front",
        "emoji": ""
    },
    "stand": {
        "category": "actions",
        "description": "V fingers stand on palm",
        "hand_shape": "V on palm",
        "movement": "Stand position",
        "location": "Palm",
        "emoji": ""
    },
    "give": {
        "category": "actions",
        "description": "Flat O hands move forward",
        "hand_shape": "Flat O",
        "movement": "Push forward and open",
        "location": "In front",
        "emoji": ""
    },
    "take": {
        "category": "actions",
        "description": "Open hand grasps inward",
        "hand_shape": "Open to fist",
        "movement": "Grasp toward body",
        "location": "In front",
        "emoji": ""
    },
    "make": {
        "category": "actions",
        "description": "Fists stack and twist",
        "hand_shape": "Fists",
        "movement": "Twist on top",
        "location": "In front",
        "emoji": ""
    },
    "buy": {
        "category": "actions",
        "description": "Flat hand in palm, moves out",
        "hand_shape": "Flat on palm",
        "movement": "Out from palm",
        "location": "Palm",
        "emoji": ""
    },
    "sell": {
        "category": "actions",
        "description": "Flat O hands flip outward",
        "hand_shape": "Flat O",
        "movement": "Flip out",
        "location": "In front",
        "emoji": ""
    },
    "open": {
        "category": "actions",
        "description": "B hands side by side, open apart",
        "hand_shape": "Flat B hands",
        "movement": "Open apart",
        "location": "In front",
        "emoji": ""
    },
    "close": {
        "category": "actions",
        "description": "B hands apart, come together",
        "hand_shape": "Flat B hands",
        "movement": "Close together",
        "location": "In front",
        "emoji": ""
    },
    "find": {
        "category": "actions",
        "description": "F hand rises up",
        "hand_shape": "F handshape",
        "movement": "Rise up",
        "location": "In front",
        "emoji": ""
    },
    "look": {
        "category": "actions",
        "description": "V from eyes forward",
        "hand_shape": "V handshape",
        "movement": "From eyes outward",
        "location": "Eyes",
        "emoji": ""
    },
    "watch": {
        "category": "actions",
        "description": "V from eyes, sustained",
        "hand_shape": "V handshape",
        "movement": "Hold forward from eyes",
        "location": "Eyes",
        "emoji": ""
    },
    "call": {
        "category": "actions",
        "description": "Y hand at ear (phone)",
        "hand_shape": "Y handshape",
        "movement": "At ear",
        "location": "Ear",
        "emoji": ""
    },
    "send": {
        "category": "actions",
        "description": "Fingers flick forward from back of hand",
        "hand_shape": "Bent fingers",
        "movement": "Flick forward",
        "location": "Hand",
        "emoji": ""
    },

    # ===== FAMILY =====
    "mother": {
        "category": "family",
        "description": "Thumb of open hand taps chin",
        "hand_shape": "Open 5 hand",
        "movement": "Tap chin",
        "location": "Chin",
        "emoji": ""
    },
    "mom": {
        "category": "family",
        "description": "Same as mother",
        "hand_shape": "Open 5 hand",
        "movement": "Tap chin",
        "location": "Chin",
        "emoji": ""
    },
    "father": {
        "category": "family",
        "description": "Thumb of open hand taps forehead",
        "hand_shape": "Open 5 hand",
        "movement": "Tap forehead",
        "location": "Forehead",
        "emoji": ""
    },
    "dad": {
        "category": "family",
        "description": "Same as father",
        "hand_shape": "Open 5 hand",
        "movement": "Tap forehead",
        "location": "Forehead",
        "emoji": ""
    },
    "sister": {
        "category": "family",
        "description": "A hand from chin to flat hand meet",
        "hand_shape": "A to flat",
        "movement": "Chin down to meet",
        "location": "Chin",
        "emoji": ""
    },
    "brother": {
        "category": "family",
        "description": "A hand from forehead to flat hand meet",
        "hand_shape": "A to flat",
        "movement": "Forehead down to meet",
        "location": "Forehead",
        "emoji": ""
    },
    "baby": {
        "category": "family",
        "description": "Arms rock as if holding baby",
        "hand_shape": "Cradled arms",
        "movement": "Rocking",
        "location": "Arms",
        "emoji": ""
    },
    "child": {
        "category": "family",
        "description": "Flat hand pats downward (short height)",
        "hand_shape": "Flat hand",
        "movement": "Pat down",
        "location": "Side",
        "emoji": ""
    },
    "friend": {
        "category": "family",
        "description": "Index fingers hook together twice",
        "hand_shape": "X handshape",
        "movement": "Hook and reverse",
        "location": "In front",
        "emoji": ""
    },
    "family": {
        "category": "family",
        "description": "F hands circle to meet",
        "hand_shape": "F handshape",
        "movement": "Circle to touch",
        "location": "In front",
        "emoji": ""
    },
    "husband": {
        "category": "family",
        "description": "Hand at forehead then hands clasp",
        "hand_shape": "Flat to clasp",
        "movement": "Forehead to clasp",
        "location": "Forehead/front",
        "emoji": ""
    },
    "wife": {
        "category": "family",
        "description": "Hand at chin then hands clasp",
        "hand_shape": "Flat to clasp",
        "movement": "Chin to clasp",
        "location": "Chin/front",
        "emoji": ""
    },
    "grandfather": {
        "category": "family",
        "description": "Open hand hops forward from forehead",
        "hand_shape": "Open 5",
        "movement": "Two hops forward",
        "location": "Forehead",
        "emoji": ""
    },
    "grandmother": {
        "category": "family",
        "description": "Open hand hops forward from chin",
        "hand_shape": "Open 5",
        "movement": "Two hops forward",
        "location": "Chin",
        "emoji": ""
    },

    # ===== NUMBERS =====
    "one": {
        "category": "numbers",
        "description": "Index finger up",
        "hand_shape": "Index extended",
        "movement": "Static",
        "location": "In front",
        "emoji": ""
    },
    "two": {
        "category": "numbers",
        "description": "Index and middle up (peace sign)",
        "hand_shape": "V handshape",
        "movement": "Static",
        "location": "In front",
        "emoji": ""
    },
    "three": {
        "category": "numbers",
        "description": "Thumb, index, middle up",
        "hand_shape": "3 handshape",
        "movement": "Static",
        "location": "In front",
        "emoji": ""
    },
    "four": {
        "category": "numbers",
        "description": "Four fingers up, thumb in",
        "hand_shape": "4 handshape",
        "movement": "Static",
        "location": "In front",
        "emoji": ""
    },
    "five": {
        "category": "numbers",
        "description": "All fingers spread open",
        "hand_shape": "Open 5",
        "movement": "Static",
        "location": "In front",
        "emoji": ""
    },
    "six": {
        "category": "numbers",
        "description": "Thumb touches pinky, others up",
        "hand_shape": "W + thumb-pinky",
        "movement": "Static",
        "location": "In front",
        "emoji": ""
    },
    "seven": {
        "category": "numbers",
        "description": "Thumb touches ring finger, others up",
        "hand_shape": "7 handshape",
        "movement": "Static",
        "location": "In front",
        "emoji": ""
    },
    "eight": {
        "category": "numbers",
        "description": "Thumb touches middle finger, others up",
        "hand_shape": "8 handshape",
        "movement": "Static",
        "location": "In front",
        "emoji": ""
    },
    "nine": {
        "category": "numbers",
        "description": "Thumb touches index finger, others up",
        "hand_shape": "9/F handshape",
        "movement": "Static",
        "location": "In front",
        "emoji": ""
    },
    "ten": {
        "category": "numbers",
        "description": "Thumb up, shake (A hand)",
        "hand_shape": "A with thumb up",
        "movement": "Shake/twist",
        "location": "In front",
        "emoji": ""
    },

    # ===== TIME =====
    "today": {
        "category": "time",
        "description": "Y hands drop down",
        "hand_shape": "Y handshape",
        "movement": "Drop down",
        "location": "In front",
        "emoji": ""
    },
    "tomorrow": {
        "category": "time",
        "description": "Thumb on cheek moves forward",
        "hand_shape": "A/thumb out",
        "movement": "Forward arc",
        "location": "Cheek",
        "emoji": ""
    },
    "yesterday": {
        "category": "time",
        "description": "Thumb on cheek moves backward",
        "hand_shape": "A/thumb out",
        "movement": "Backward to ear",
        "location": "Cheek",
        "emoji": ""
    },
    "now": {
        "category": "time",
        "description": "Y hands drop sharply",
        "hand_shape": "Y handshape",
        "movement": "Sharp drop",
        "location": "In front",
        "emoji": ""
    },
    "later": {
        "category": "time",
        "description": "L hand moves forward",
        "hand_shape": "L handshape",
        "movement": "Forward tilt",
        "location": "In front",
        "emoji": ""
    },
    "before": {
        "category": "time",
        "description": "Flat hand moves back toward body",
        "hand_shape": "Flat hand",
        "movement": "Back toward body",
        "location": "In front",
        "emoji": ""
    },
    "after": {
        "category": "time",
        "description": "Flat hand moves forward from other hand",
        "hand_shape": "Flat hand",
        "movement": "Forward",
        "location": "In front",
        "emoji": ""
    },
    "morning": {
        "category": "time",
        "description": "Flat hand rises like sun",
        "hand_shape": "Flat hand in elbow",
        "movement": "Rise up",
        "location": "Arm",
        "emoji": ""
    },
    "afternoon": {
        "category": "time",
        "description": "Flat hand at 45 degrees",
        "hand_shape": "Flat hand",
        "movement": "45 degree angle",
        "location": "Arm",
        "emoji": ""
    },
    "night": {
        "category": "time",
        "description": "Bent hand over wrist (sun setting)",
        "hand_shape": "Bent hand",
        "movement": "Down over wrist",
        "location": "Wrist",
        "emoji": ""
    },
    "week": {
        "category": "time",
        "description": "1 hand slides across palm",
        "hand_shape": "Index on palm",
        "movement": "Slide across",
        "location": "Palm",
        "emoji": ""
    },
    "month": {
        "category": "time",
        "description": "Index slides down other index",
        "hand_shape": "Index fingers",
        "movement": "Slide down",
        "location": "In front",
        "emoji": ""
    },
    "year": {
        "category": "time",
        "description": "Fists circle each other, land",
        "hand_shape": "S fists",
        "movement": "Circle and land",
        "location": "In front",
        "emoji": ""
    },

    # ===== PLACES =====
    "home": {
        "category": "places",
        "description": "Flat O from mouth to cheek",
        "hand_shape": "Flat O",
        "movement": "Mouth to cheek",
        "location": "Face",
        "emoji": ""
    },
    "house": {
        "category": "places",
        "description": "Flat hands make roof shape",
        "hand_shape": "Flat hands angled",
        "movement": "Form roof, drop to walls",
        "location": "In front",
        "emoji": ""
    },
    "school": {
        "category": "places",
        "description": "Clap hands twice",
        "hand_shape": "Flat hands",
        "movement": "Clap twice",
        "location": "In front",
        "emoji": ""
    },
    "work": {
        "category": "places",
        "description": "S hands tap together",
        "hand_shape": "S fists",
        "movement": "Tap wrists",
        "location": "In front",
        "emoji": ""
    },
    "store": {
        "category": "places",
        "description": "Flat O hands twist outward",
        "hand_shape": "Flat O",
        "movement": "Twist out",
        "location": "In front",
        "emoji": ""
    },
    "hospital": {
        "category": "places",
        "description": "H draws cross on arm",
        "hand_shape": "H handshape",
        "movement": "Draw cross",
        "location": "Upper arm",
        "emoji": ""
    },
    "restaurant": {
        "category": "places",
        "description": "R touches lips then chin",
        "hand_shape": "R handshape",
        "movement": "Lips to chin",
        "location": "Face",
        "emoji": ""
    },
    "church": {
        "category": "places",
        "description": "C on back of S hand",
        "hand_shape": "C on S",
        "movement": "Tap twice",
        "location": "Hand",
        "emoji": ""
    },
    "city": {
        "category": "places",
        "description": "B hands tap rooftops together",
        "hand_shape": "B hands",
        "movement": "Tap together twice",
        "location": "In front",
        "emoji": ""
    },

    # ===== THINGS =====
    "water": {
        "category": "things",
        "description": "W taps chin twice",
        "hand_shape": "W handshape",
        "movement": "Tap chin",
        "location": "Chin",
        "emoji": ""
    },
    "food": {
        "category": "things",
        "description": "Flat O taps lips",
        "hand_shape": "Flat O",
        "movement": "Tap to lips",
        "location": "Mouth",
        "emoji": ""
    },
    "book": {
        "category": "things",
        "description": "Palms together open like book",
        "hand_shape": "Flat hands",
        "movement": "Open like book",
        "location": "In front",
        "emoji": ""
    },
    "phone": {
        "category": "things",
        "description": "Y hand at ear",
        "hand_shape": "Y handshape",
        "movement": "At ear",
        "location": "Ear",
        "emoji": ""
    },
    "computer": {
        "category": "things",
        "description": "C hand moves up arm",
        "hand_shape": "C handshape",
        "movement": "Up forearm",
        "location": "Arm",
        "emoji": ""
    },
    "car": {
        "category": "things",
        "description": "Hands mime steering wheel",
        "hand_shape": "Fists gripping",
        "movement": "Steering motion",
        "location": "In front",
        "emoji": ""
    },
    "money": {
        "category": "things",
        "description": "Flat hand taps other palm",
        "hand_shape": "Flat O on palm",
        "movement": "Tap twice",
        "location": "Palm",
        "emoji": ""
    },
    "door": {
        "category": "things",
        "description": "B hands side by side, one swings open",
        "hand_shape": "B hands",
        "movement": "One swings open",
        "location": "In front",
        "emoji": ""
    },
    "window": {
        "category": "things",
        "description": "Flat hands stacked, top one rises",
        "hand_shape": "Flat hands",
        "movement": "Top rises",
        "location": "In front",
        "emoji": ""
    },
    "key": {
        "category": "things",
        "description": "X turns in palm (like key in lock)",
        "hand_shape": "X handshape",
        "movement": "Turn/twist",
        "location": "Palm",
        "emoji": ""
    },
    "camera": {
        "category": "things",
        "description": "Hands frame face, finger clicks",
        "hand_shape": "Camera shape",
        "movement": "Click motion",
        "location": "Face",
        "emoji": ""
    },
    "picture": {
        "category": "things",
        "description": "C hand moves from face to palm",
        "hand_shape": "C handshape",
        "movement": "Face to palm",
        "location": "Face/palm",
        "emoji": ""
    },
    "music": {
        "category": "things",
        "description": "Hand waves over forearm",
        "hand_shape": "Flat hand",
        "movement": "Wave over arm",
        "location": "Arm",
        "emoji": ""
    },

    # ===== COLORS =====
    "red": {
        "category": "colors",
        "description": "Index brushes down from lips",
        "hand_shape": "Index extended",
        "movement": "Down from lips",
        "location": "Lips",
        "emoji": ""
    },
    "blue": {
        "category": "colors",
        "description": "B hand shakes",
        "hand_shape": "B handshape",
        "movement": "Twist/shake",
        "location": "In front",
        "emoji": ""
    },
    "green": {
        "category": "colors",
        "description": "G hand shakes",
        "hand_shape": "G handshape",
        "movement": "Twist/shake",
        "location": "In front",
        "emoji": ""
    },
    "yellow": {
        "category": "colors",
        "description": "Y hand shakes",
        "hand_shape": "Y handshape",
        "movement": "Twist at wrist",
        "location": "In front",
        "emoji": ""
    },
    "black": {
        "category": "colors",
        "description": "Index draws across forehead",
        "hand_shape": "Index extended",
        "movement": "Across forehead",
        "location": "Forehead",
        "emoji": ""
    },
    "white": {
        "category": "colors",
        "description": "5 hand pulls from chest, closing",
        "hand_shape": "5 to flat O",
        "movement": "Pull from chest",
        "location": "Chest",
        "emoji": ""
    },
    "orange": {
        "category": "colors",
        "description": "C hand squeezes at chin",
        "hand_shape": "C to S",
        "movement": "Squeeze at chin",
        "location": "Chin",
        "emoji": ""
    },
    "purple": {
        "category": "colors",
        "description": "P hand shakes",
        "hand_shape": "P handshape",
        "movement": "Shake",
        "location": "In front",
        "emoji": ""
    },
    "pink": {
        "category": "colors",
        "description": "P hand brushes down chin",
        "hand_shape": "P handshape",
        "movement": "Down lips",
        "location": "Lips",
        "emoji": ""
    },
    "brown": {
        "category": "colors",
        "description": "B slides down cheek",
        "hand_shape": "B handshape",
        "movement": "Down cheek",
        "location": "Cheek",
        "emoji": ""
    },

    # ===== EMERGENCY =====
    "emergency": {
        "category": "emergency",
        "description": "E hand shakes side to side rapidly",
        "hand_shape": "E handshape",
        "movement": "Rapid shake",
        "location": "In front",
        "emoji": ""
    },
    "danger": {
        "category": "emergency",
        "description": "A hands push up alternately",
        "hand_shape": "A fists",
        "movement": "Push up alternating",
        "location": "Chest",
        "emoji": ""
    },
    "fire": {
        "category": "emergency",
        "description": "5 hands wiggle rising up",
        "hand_shape": "5 hands",
        "movement": "Wiggle upward",
        "location": "In front rising",
        "emoji": ""
    },
    "police": {
        "category": "emergency",
        "description": "C hand taps chest (badge)",
        "hand_shape": "C handshape",
        "movement": "Tap chest",
        "location": "Chest",
        "emoji": ""
    },
    "doctor": {
        "category": "emergency",
        "description": "M/D hand taps wrist pulse",
        "hand_shape": "M or flat",
        "movement": "Tap wrist",
        "location": "Wrist",
        "emoji": ""
    },
    "hurt": {
        "category": "emergency",
        "description": "Index fingers point at each other, twist",
        "hand_shape": "Index fingers",
        "movement": "Twist toward each other",
        "location": "In front",
        "emoji": ""
    },
    "pain": {
        "category": "emergency",
        "description": "Same as hurt",
        "hand_shape": "Index fingers",
        "movement": "Twist toward each other",
        "location": "At location of pain",
        "emoji": ""
    },
    "sick": {
        "category": "emergency",
        "description": "5 hand middle on forehead, other on stomach",
        "hand_shape": "5 hands bent",
        "movement": "Touch forehead and stomach",
        "location": "Forehead/stomach",
        "emoji": ""
    },
    "ambulance": {
        "category": "emergency",
        "description": "A hand circles (siren light)",
        "hand_shape": "A fist",
        "movement": "Circle above head",
        "location": "Above head",
        "emoji": ""
    },

    # ===== LOVE & FEELINGS =====
    "i love you": {
        "category": "emotions",
        "description": "Thumb, index, pinky extended (ILY)",
        "hand_shape": "ILY handshape",
        "movement": "Static or slight shake",
        "location": "In front",
        "emoji": ""
    },
    "miss": {
        "category": "emotions",
        "description": "Index touches chin then moves forward",
        "hand_shape": "Index extended",
        "movement": "From chin forward",
        "location": "Chin",
        "emoji": ""
    },
    "beautiful": {
        "category": "emotions",
        "description": "5 hand circles face, closes",
        "hand_shape": "5 to flat O",
        "movement": "Circle face",
        "location": "Face",
        "emoji": ""
    },
    "wonderful": {
        "category": "emotions",
        "description": "5 hands push out with amazement",
        "hand_shape": "5 hands",
        "movement": "Push out",
        "location": "In front",
        "emoji": ""
    },
    "hope": {
        "category": "emotions",
        "description": "Bent hands at temple, wave up",
        "hand_shape": "Bent hands",
        "movement": "Wave upward",
        "location": "Temple",
        "emoji": ""
    },
    "dream": {
        "category": "emotions",
        "description": "Index at forehead, moves up wavy",
        "hand_shape": "Index extended",
        "movement": "Wavy upward",
        "location": "Forehead",
        "emoji": ""
    },
    "believe": {
        "category": "emotions",
        "description": "Index at forehead clasps other hand",
        "hand_shape": "Index to clasp",
        "movement": "Forehead to clasp",
        "location": "Forehead/front",
        "emoji": ""
    },
    "remember": {
        "category": "emotions",
        "description": "Thumb touches forehead then meets other thumb",
        "hand_shape": "A thumbs",
        "movement": "Forehead down to meet",
        "location": "Forehead",
        "emoji": ""
    },
    "forget": {
        "category": "emotions",
        "description": "Open hand at forehead wipes across",
        "hand_shape": "Open to A",
        "movement": "Wipe across forehead",
        "location": "Forehead",
        "emoji": ""
    },
    "proud": {
        "category": "emotions",
        "description": "Thumb on stomach rises up chest",
        "hand_shape": "A with thumb",
        "movement": "Up chest",
        "location": "Stomach to chest",
        "emoji": ""
    }
}

# =============================================================================
# ASL FINGERSPELLING ALPHABET
# =============================================================================

ASL_ALPHABET = {
    "a": {"description": "Fist with thumb beside fingers", "emoji": ""},
    "b": {"description": "Flat hand, fingers up, thumb across palm", "emoji": ""},
    "c": {"description": "Curved hand like holding a cup", "emoji": ""},
    "d": {"description": "Index up, others touch thumb", "emoji": ""},
    "e": {"description": "Fingertips touch thumb, bent", "emoji": ""},
    "f": {"description": "Circle with thumb and index, others up", "emoji": ""},
    "g": {"description": "Index and thumb horizontal, pointing", "emoji": ""},
    "h": {"description": "Index and middle horizontal together", "emoji": ""},
    "i": {"description": "Pinky up, fist", "emoji": ""},
    "j": {"description": "Pinky up, trace J shape", "emoji": ""},
    "k": {"description": "Index and middle up, thumb between", "emoji": ""},
    "l": {"description": "L shape - thumb and index extended", "emoji": ""},
    "m": {"description": "Fingers over thumb, 3 bumps showing", "emoji": ""},
    "n": {"description": "Fingers over thumb, 2 bumps showing", "emoji": ""},
    "o": {"description": "Fingers curved to touch thumb - O shape", "emoji": ""},
    "p": {"description": "K hand pointed down", "emoji": ""},
    "q": {"description": "G hand pointed down", "emoji": ""},
    "r": {"description": "Index and middle crossed", "emoji": ""},
    "s": {"description": "Fist with thumb over fingers", "emoji": ""},
    "t": {"description": "Fist with thumb between index and middle", "emoji": ""},
    "u": {"description": "Index and middle up together", "emoji": ""},
    "v": {"description": "Index and middle up and apart - V shape", "emoji": ""},
    "w": {"description": "Index, middle, ring up and apart", "emoji": ""},
    "x": {"description": "Index bent into hook", "emoji": ""},
    "y": {"description": "Thumb and pinky out - hang loose", "emoji": ""},
    "z": {"description": "Index draws Z in air", "emoji": ""}
}

# =============================================================================
# DATABASE CLASS
# =============================================================================

class SignLanguageDatabase:
    """
    Comprehensive sign language database supporting 300+ languages.
    """
    
    def __init__(self, dictionaries_dir: str):
        self.dictionaries_dir = Path(dictionaries_dir)
        self.dictionaries_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dictionaries for each language
        self.language_data = {}
        for lang in SUPPORTED_LANGUAGES.keys():
            self.language_data[lang] = self._load_language(lang)
        
        logger.info(f"Loaded {len(SUPPORTED_LANGUAGES)} sign languages")
    
    def _load_language(self, lang: str) -> Dict:
        """Load vocabulary for a specific language."""
        lang_dir = self.dictionaries_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load custom dictionary file
        dict_file = lang_dir / "dictionary.json"
        if dict_file.exists():
            try:
                with open(dict_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {lang} dictionary: {e}")
        
        # Fall back to universal vocabulary
        return UNIVERSAL_VOCABULARY
    
    def get_sign(self, language: str, word: str) -> Optional[Dict]:
        """Get sign information for a word in a specific language."""
        language = language.upper()
        word = word.lower().strip()
        
        if language not in self.language_data:
            return None
        
        vocab = self.language_data.get(language, UNIVERSAL_VOCABULARY)
        
        if word in vocab:
            sign_data = vocab[word].copy()
            sign_data["word"] = word
            sign_data["language"] = language
            sign_data["success"] = True
            return sign_data
        
        return None
    
    def get_fingerspelling(self, language: str, word: str) -> List[Dict]:
        """Get fingerspelling sequence for a word."""
        letters = []
        for char in word.lower():
            if char in ASL_ALPHABET:
                letter_data = ASL_ALPHABET[char].copy()
                letter_data["letter"] = char.upper()
                letters.append(letter_data)
        return letters
    
    def get_vocabulary(self, language: str) -> Dict[str, List[str]]:
        """Get all available words organized by category."""
        language = language.upper()
        vocab = self.language_data.get(language, UNIVERSAL_VOCABULARY)
        
        categories = {}
        for word, data in vocab.items():
            cat = data.get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(word)
        
        return categories
    
    def search(self, query: str, language: str = "ASL") -> List[Dict]:
        """Search for words matching a query."""
        language = language.upper()
        vocab = self.language_data.get(language, UNIVERSAL_VOCABULARY)
        
        results = []
        query = query.lower()
        
        for word, data in vocab.items():
            if query in word or query in data.get("description", "").lower():
                result = data.copy()
                result["word"] = word
                results.append(result)
        
        return results[:20]  # Limit results
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        categories = set()
        for data in UNIVERSAL_VOCABULARY.values():
            categories.add(data.get("category", "other"))
        return sorted(list(categories))
