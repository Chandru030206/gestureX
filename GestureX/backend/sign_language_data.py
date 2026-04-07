"""
Multi-Language Sign Language Data Module
Contains gesture descriptions and audio explanations for words across different sign languages.
Each sign language has its OWN unique gesture representation.
"""

# Comprehensive sign language dictionary
# Structure: { language: { word: { description, gesture_steps, audio_text, emoji } } }

SIGN_LANGUAGE_DATA = {
    "ASL": {
        "name": "American Sign Language",
        "country": "United States",
        "hello": {
            "description": "Wave with flat hand from forehead",
            "gesture_steps": [
                "Hold your dominant hand flat with fingers together",
                "Place your hand near your forehead (like a salute)",
                "Move your hand outward and slightly away from your head",
                "The motion is similar to a military salute moving outward"
            ],
            "audio_text": "In American Sign Language, hello is signed by holding a flat hand near your forehead, similar to a salute, and then moving it outward away from your head. This gesture represents acknowledging someone's presence.",
            "emoji": "👋",
            "hand_shape": "Flat B hand",
            "movement": "Outward from forehead"
        },
        "thank you": {
            "description": "Flat hand from chin moving outward",
            "gesture_steps": [
                "Touch your fingertips to your chin",
                "Move your hand forward and slightly down",
                "Palm faces upward as it moves away",
                "Like blowing a kiss of gratitude"
            ],
            "audio_text": "In American Sign Language, thank you is signed by touching your fingertips to your chin and then moving your hand forward and downward with palm facing up. It's like sending gratitude from your mouth to the other person.",
            "emoji": "🙏",
            "hand_shape": "Flat hand",
            "movement": "From chin outward"
        },
        "please": {
            "description": "Circular motion on chest",
            "gesture_steps": [
                "Place your flat hand on your chest",
                "Make a circular rubbing motion",
                "Keep the motion smooth and continuous",
                "Express sincerity with facial expression"
            ],
            "audio_text": "In American Sign Language, please is signed by placing your flat hand on your chest and making a circular rubbing motion. This gesture shows sincerity and politeness in your request.",
            "emoji": "🤲",
            "hand_shape": "Flat hand on chest",
            "movement": "Circular rubbing"
        },
        "sorry": {
            "description": "Fist circles on chest",
            "gesture_steps": [
                "Make a fist with your dominant hand",
                "Place it on your chest",
                "Rub in a circular motion",
                "Show remorseful facial expression"
            ],
            "audio_text": "In American Sign Language, sorry is signed by making a fist and rubbing it in a circular motion on your chest. The circular motion combined with a remorseful expression conveys sincere apology.",
            "emoji": "😔",
            "hand_shape": "A hand (fist)",
            "movement": "Circular on chest"
        },
        "help": {
            "description": "Thumbs up on flat palm, lift up",
            "gesture_steps": [
                "Make a thumbs-up with your dominant hand",
                "Place it on your flat non-dominant palm",
                "Lift both hands upward together",
                "Like someone lifting you up"
            ],
            "audio_text": "In American Sign Language, help is signed by placing a thumbs-up on your flat palm and lifting both hands upward. This represents someone being lifted up or supported by another person.",
            "emoji": "🆘",
            "hand_shape": "Thumbs up on flat palm",
            "movement": "Upward lift"
        },
        "yes": {
            "description": "Fist nods up and down",
            "gesture_steps": [
                "Make a fist with your dominant hand",
                "Hold it in front of you",
                "Move it up and down like nodding",
                "Repeat the nodding motion 2-3 times"
            ],
            "audio_text": "In American Sign Language, yes is signed by making a fist and moving it up and down, mimicking a nodding head. This simple gesture clearly indicates agreement or affirmation.",
            "emoji": "✅",
            "hand_shape": "S hand (fist)",
            "movement": "Nodding motion"
        },
        "no": {
            "description": "Index and middle finger snap to thumb",
            "gesture_steps": [
                "Extend your index and middle fingers",
                "Hold your thumb out",
                "Snap your fingers together to meet the thumb",
                "Like closing a beak or pinching motion"
            ],
            "audio_text": "In American Sign Language, no is signed by snapping your extended index and middle fingers together with your thumb, like closing a beak. This quick motion represents the concept of negation.",
            "emoji": "❌",
            "hand_shape": "Modified N hand",
            "movement": "Snapping to thumb"
        },
        "love": {
            "description": "Cross arms over chest in hug",
            "gesture_steps": [
                "Cross both arms over your chest",
                "Hands rest on opposite shoulders",
                "Like giving yourself a hug",
                "Hold briefly with warm expression"
            ],
            "audio_text": "In American Sign Language, love is signed by crossing your arms over your chest as if hugging yourself. This beautiful gesture symbolizes embracing someone with affection.",
            "emoji": "❤️",
            "hand_shape": "Crossed arms",
            "movement": "Hugging motion"
        },
        "water": {
            "description": "W hand taps chin",
            "gesture_steps": [
                "Make a W shape with three fingers extended",
                "Tap your index finger on your chin",
                "Tap 2-3 times",
                "Represents the W in water"
            ],
            "audio_text": "In American Sign Language, water is signed by making a W hand shape with three fingers extended and tapping your chin. The W represents the first letter of water.",
            "emoji": "💧",
            "hand_shape": "W hand",
            "movement": "Tapping chin"
        },
        "food": {
            "description": "Fingers tap lips repeatedly",
            "gesture_steps": [
                "Bring your fingers together in a flat O shape",
                "Tap your fingertips to your lips",
                "Repeat the tapping motion",
                "Like putting food in your mouth"
            ],
            "audio_text": "In American Sign Language, food is signed by bringing your fingertips together and tapping them to your lips repeatedly. This mimics the action of eating or putting food in your mouth.",
            "emoji": "🍽️",
            "hand_shape": "Flat O hand",
            "movement": "Tapping lips"
        }
    },
    
    "BSL": {
        "name": "British Sign Language",
        "country": "United Kingdom",
        "hello": {
            "description": "Wave open palm side to side",
            "gesture_steps": [
                "Hold your hand up with palm facing forward",
                "Keep fingers together and extended",
                "Wave your hand side to side",
                "Maintain a friendly facial expression"
            ],
            "audio_text": "In British Sign Language, hello is expressed by waving an open palm side to side at head height. This is a natural waving gesture that mirrors the common greeting wave, making it universally recognizable.",
            "emoji": "👋",
            "hand_shape": "Open palm",
            "movement": "Side to side wave"
        },
        "thank you": {
            "description": "Flat hand from chin moving forward",
            "gesture_steps": [
                "Place your flat hand under your chin",
                "Move your hand forward and slightly down",
                "Palm faces down as it moves",
                "Single smooth movement"
            ],
            "audio_text": "In British Sign Language, thank you is signed by placing a flat hand under your chin and moving it forward and downward. This gesture represents sending your thanks outward to the other person.",
            "emoji": "🙏",
            "hand_shape": "Flat hand",
            "movement": "Forward from chin"
        },
        "please": {
            "description": "Flat hand circles on chest",
            "gesture_steps": [
                "Place your flat hand on your upper chest",
                "Make a small circular motion",
                "Keep movement gentle and polite",
                "Maintain pleasant expression"
            ],
            "audio_text": "In British Sign Language, please is signed by placing your flat hand on your chest and making a gentle circular motion. This shows politeness and sincerity in British deaf culture.",
            "emoji": "🤲",
            "hand_shape": "Flat hand",
            "movement": "Circular on chest"
        },
        "sorry": {
            "description": "Fist rotates on chest",
            "gesture_steps": [
                "Make a fist with your dominant hand",
                "Place it on your chest",
                "Rotate in a circular motion",
                "Show apologetic facial expression"
            ],
            "audio_text": "In British Sign Language, sorry is signed by making a fist and rotating it in a circular motion on your chest. The rotation combined with a sorry expression conveys genuine apology.",
            "emoji": "😔",
            "hand_shape": "Closed fist",
            "movement": "Rotating on chest"
        },
        "help": {
            "description": "One hand lifts the other",
            "gesture_steps": [
                "Hold one hand flat, palm up",
                "Place your other fist on top",
                "Lift both hands together upward",
                "Represents giving support"
            ],
            "audio_text": "In British Sign Language, help is signed by placing a closed fist on an open palm and lifting both hands upward together. This represents one person supporting or lifting up another.",
            "emoji": "🆘",
            "hand_shape": "Fist on palm",
            "movement": "Lifting upward"
        },
        "yes": {
            "description": "Head nod with fist movement",
            "gesture_steps": [
                "Make a fist with your hand",
                "Move it up and down",
                "Like your head nodding",
                "Can be accompanied by actual nod"
            ],
            "audio_text": "In British Sign Language, yes is commonly expressed through a nodding motion, either with the head alone or accompanied by a fist moving up and down. This natural gesture is easily understood.",
            "emoji": "✅",
            "hand_shape": "Fist",
            "movement": "Nodding motion"
        },
        "no": {
            "description": "Index finger wags side to side",
            "gesture_steps": [
                "Extend your index finger",
                "Hold it upright",
                "Wag it side to side",
                "Like saying 'no no' with your finger"
            ],
            "audio_text": "In British Sign Language, no is often signed by wagging your index finger side to side, similar to the universal 'no no' gesture. This can also be accompanied by shaking your head.",
            "emoji": "❌",
            "hand_shape": "Index finger",
            "movement": "Wagging side to side"
        },
        "love": {
            "description": "Arms cross over heart",
            "gesture_steps": [
                "Cross both arms over your chest",
                "Place hands on opposite shoulders",
                "Like hugging yourself",
                "Express warmth in your face"
            ],
            "audio_text": "In British Sign Language, love is signed by crossing your arms over your chest in a hugging motion. This gesture symbolizes embracing someone dear to you.",
            "emoji": "❤️",
            "hand_shape": "Crossed arms",
            "movement": "Hugging self"
        },
        "water": {
            "description": "W hand shape taps chin",
            "gesture_steps": [
                "Form a W with three fingers",
                "Tap the W on your chin",
                "Repeat 2-3 times",
                "W represents water"
            ],
            "audio_text": "In British Sign Language, water is signed using a W hand shape tapped on the chin. The W represents the first letter of the word water.",
            "emoji": "💧",
            "hand_shape": "W shape",
            "movement": "Tapping chin"
        },
        "food": {
            "description": "Hand moves to mouth repeatedly",
            "gesture_steps": [
                "Bunch your fingers together",
                "Move them toward your mouth",
                "Repeat the motion",
                "Mimics eating action"
            ],
            "audio_text": "In British Sign Language, food is signed by bringing bunched fingers toward your mouth repeatedly, mimicking the action of eating. This intuitive gesture is easy to understand.",
            "emoji": "🍽️",
            "hand_shape": "Bunched fingers",
            "movement": "To mouth"
        }
    },
    
    "ISL": {
        "name": "Indian Sign Language",
        "country": "India",
        "hello": {
            "description": "Namaste gesture with slight bow",
            "gesture_steps": [
                "Bring both palms together in front of chest",
                "Fingers pointing upward",
                "Slight bow of the head",
                "Traditional Indian greeting position"
            ],
            "audio_text": "In Indian Sign Language, hello is expressed through the traditional Namaste gesture. Both palms are pressed together in front of the chest with fingers pointing upward, accompanied by a slight bow of the head. This respectful greeting is deeply rooted in Indian culture.",
            "emoji": "🙏",
            "hand_shape": "Palms together",
            "movement": "Slight bow"
        },
        "thank you": {
            "description": "Namaste gesture with deeper bow",
            "gesture_steps": [
                "Press palms together at chest level",
                "Fingers point upward",
                "Bow your head more deeply",
                "Express gratitude through expression"
            ],
            "audio_text": "In Indian Sign Language, thank you is also expressed through the Namaste gesture but with a deeper bow to show greater respect and gratitude. The depth of the bow indicates the level of thankfulness.",
            "emoji": "🙏",
            "hand_shape": "Palms pressed together",
            "movement": "Deep bow"
        },
        "please": {
            "description": "Palms together with pleading expression",
            "gesture_steps": [
                "Join palms together",
                "Hold in front of chest",
                "Slight forward tilt",
                "Pleading or requesting expression"
            ],
            "audio_text": "In Indian Sign Language, please is signed with joined palms held in front of the chest, similar to a prayer position, with a slight forward tilt and a requesting facial expression. This shows humility in making a request.",
            "emoji": "🤲",
            "hand_shape": "Joined palms",
            "movement": "Forward tilt"
        },
        "sorry": {
            "description": "Hand on heart with apologetic bow",
            "gesture_steps": [
                "Place your right hand on your heart",
                "Bow your head slightly",
                "Show regretful expression",
                "May add slight hand motion outward"
            ],
            "audio_text": "In Indian Sign Language, sorry is expressed by placing your hand on your heart and bowing your head slightly with a regretful expression. This heartfelt gesture shows sincere apology in Indian deaf culture.",
            "emoji": "😔",
            "hand_shape": "Hand on heart",
            "movement": "Bow head"
        },
        "help": {
            "description": "One hand supports the other, lifting up",
            "gesture_steps": [
                "Hold one hand flat, palm up",
                "Place other hand on top",
                "Lift both hands upward",
                "Represents giving assistance"
            ],
            "audio_text": "In Indian Sign Language, help is signed by placing one hand on top of another and lifting both upward together. This represents the act of supporting or assisting someone in need.",
            "emoji": "🆘",
            "hand_shape": "Stacked hands",
            "movement": "Lifting upward"
        },
        "yes": {
            "description": "Head nod with hand gesture",
            "gesture_steps": [
                "Nod your head up and down",
                "Can add fist movement",
                "Clear affirmative expression",
                "Single or multiple nods"
            ],
            "audio_text": "In Indian Sign Language, yes is typically expressed through a head nod, often accompanied by a fist moving up and down. A clear affirmative facial expression reinforces the meaning.",
            "emoji": "✅",
            "hand_shape": "Optional fist",
            "movement": "Nodding"
        },
        "no": {
            "description": "Head shake with hand wave",
            "gesture_steps": [
                "Shake your head side to side",
                "Wave hand in front, palm out",
                "Negative facial expression",
                "Clear dismissive motion"
            ],
            "audio_text": "In Indian Sign Language, no is expressed by shaking the head side to side, often accompanied by waving the hand with palm facing outward. This combined gesture clearly indicates negation.",
            "emoji": "❌",
            "hand_shape": "Open palm",
            "movement": "Head shake and wave"
        },
        "love": {
            "description": "Hands crossed over heart",
            "gesture_steps": [
                "Cross both hands over your heart",
                "Press gently against chest",
                "Warm, loving expression",
                "May include slight rocking"
            ],
            "audio_text": "In Indian Sign Language, love is signed by crossing both hands over your heart and pressing them against your chest. This gesture, combined with a warm expression, conveys deep affection.",
            "emoji": "❤️",
            "hand_shape": "Crossed hands",
            "movement": "On heart"
        },
        "water": {
            "description": "Drinking motion with cupped hand",
            "gesture_steps": [
                "Cup your hand",
                "Bring it to your lips",
                "Tilt like drinking",
                "Repeat motion if needed"
            ],
            "audio_text": "In Indian Sign Language, water is signed by cupping your hand and bringing it to your lips in a drinking motion. This intuitive gesture mimics the act of drinking water.",
            "emoji": "💧",
            "hand_shape": "Cupped hand",
            "movement": "Drinking motion"
        },
        "food": {
            "description": "Fingers to mouth eating motion",
            "gesture_steps": [
                "Bunch fingertips together",
                "Move toward mouth repeatedly",
                "Like picking up food to eat",
                "Natural eating gesture"
            ],
            "audio_text": "In Indian Sign Language, food is signed by bunching your fingertips together and moving them toward your mouth repeatedly, mimicking the traditional Indian way of eating with hands.",
            "emoji": "🍽️",
            "hand_shape": "Bunched fingers",
            "movement": "To mouth"
        }
    },
    
    "JSL": {
        "name": "Japanese Sign Language",
        "country": "Japan",
        "hello": {
            "description": "Bow with hands at sides or together",
            "gesture_steps": [
                "Stand with hands at your sides or together",
                "Bow from the waist",
                "Depth of bow shows respect level",
                "Maintain formal posture"
            ],
            "audio_text": "In Japanese Sign Language, hello is expressed through a bow, reflecting the important cultural practice of bowing in Japan. The hands may be at the sides or together, and the depth of the bow indicates the level of respect.",
            "emoji": "🙇",
            "hand_shape": "Hands at sides or together",
            "movement": "Bowing"
        },
        "thank you": {
            "description": "Deep bow with hands together",
            "gesture_steps": [
                "Place hands together in front",
                "Bow deeply from the waist",
                "Hold the bow briefly",
                "Rise slowly with gratitude"
            ],
            "audio_text": "In Japanese Sign Language, thank you is expressed through a deep bow with hands placed together. The depth and duration of the bow conveys the level of gratitude, reflecting Japanese cultural values of respect.",
            "emoji": "🙏",
            "hand_shape": "Hands together",
            "movement": "Deep bow"
        },
        "please": {
            "description": "Slight bow with hands pressed together",
            "gesture_steps": [
                "Press palms together",
                "Hold in front of chest",
                "Slight bow forward",
                "Polite, requesting expression"
            ],
            "audio_text": "In Japanese Sign Language, please is signed with palms pressed together and a slight bow forward. This polite gesture reflects the Japanese cultural emphasis on courtesy when making requests.",
            "emoji": "🤲",
            "hand_shape": "Palms together",
            "movement": "Slight bow"
        },
        "sorry": {
            "description": "Deep bow with hands at sides",
            "gesture_steps": [
                "Keep hands at your sides",
                "Bow deeply from the waist",
                "Hold the bow to show sincerity",
                "Apologetic facial expression"
            ],
            "audio_text": "In Japanese Sign Language, sorry is expressed through a deep bow with hands at your sides. The deeper and longer the bow, the more sincere the apology. This reflects the Japanese cultural practice of bowing to apologize.",
            "emoji": "😔",
            "hand_shape": "Hands at sides",
            "movement": "Deep bow"
        },
        "help": {
            "description": "One hand pushes up the other",
            "gesture_steps": [
                "Hold one fist with palm facing up",
                "Use other hand to push it upward",
                "Lifting motion",
                "Represents giving support"
            ],
            "audio_text": "In Japanese Sign Language, help is signed by using one hand to push the other hand upward in a lifting motion. This represents the concept of supporting or assisting someone.",
            "emoji": "🆘",
            "hand_shape": "One hand lifting other",
            "movement": "Pushing upward"
        },
        "yes": {
            "description": "Nod with possible hand confirmation",
            "gesture_steps": [
                "Nod your head clearly",
                "May add hand gesture",
                "Affirmative expression",
                "Can be subtle or pronounced"
            ],
            "audio_text": "In Japanese Sign Language, yes is typically expressed through a clear head nod, which may be accompanied by a hand gesture. The Japanese cultural tendency toward subtlety may make the nod less pronounced than in other cultures.",
            "emoji": "✅",
            "hand_shape": "Optional",
            "movement": "Head nod"
        },
        "no": {
            "description": "Hand waves in front of face",
            "gesture_steps": [
                "Hold hand in front of face",
                "Wave side to side",
                "Palm facing outward",
                "May shake head slightly"
            ],
            "audio_text": "In Japanese Sign Language, no is often signed by waving the hand in front of the face with the palm facing outward. This gesture, common in Japanese culture, politely indicates refusal or negation.",
            "emoji": "❌",
            "hand_shape": "Open hand",
            "movement": "Waving in front of face"
        },
        "love": {
            "description": "Hands form heart shape",
            "gesture_steps": [
                "Bring both hands together",
                "Form a heart shape with fingers",
                "Hold in front of chest",
                "Warm, loving expression"
            ],
            "audio_text": "In Japanese Sign Language, love can be expressed by forming a heart shape with both hands held in front of the chest. This modern gesture has become popular in Japanese deaf culture.",
            "emoji": "❤️",
            "hand_shape": "Heart shape",
            "movement": "Held at chest"
        },
        "water": {
            "description": "Index finger draws water droplet",
            "gesture_steps": [
                "Extend index finger",
                "Draw a droplet shape in the air",
                "Or make drinking motion",
                "Clear, simple gesture"
            ],
            "audio_text": "In Japanese Sign Language, water is signed by drawing a water droplet shape in the air with your index finger, or by making a drinking motion. Both gestures clearly represent the concept of water.",
            "emoji": "💧",
            "hand_shape": "Index finger or drinking",
            "movement": "Droplet or drinking"
        },
        "food": {
            "description": "Chopsticks motion to mouth",
            "gesture_steps": [
                "Hold fingers like chopsticks",
                "Move toward mouth",
                "Mimics eating with chopsticks",
                "Distinctly Japanese gesture"
            ],
            "audio_text": "In Japanese Sign Language, food is signed by mimicking the motion of using chopsticks to eat, bringing the hand toward the mouth. This culturally specific gesture reflects Japanese dining customs.",
            "emoji": "🍽️",
            "hand_shape": "Chopstick hold",
            "movement": "To mouth"
        }
    },
    
    "AUSLAN": {
        "name": "Australian Sign Language",
        "country": "Australia",
        "hello": {
            "description": "Wave with open hand",
            "gesture_steps": [
                "Hold hand up at shoulder height",
                "Open palm facing outward",
                "Wave side to side",
                "Friendly expression"
            ],
            "audio_text": "In Australian Sign Language, also known as Auslan, hello is signed with a friendly wave of the open hand. The palm faces outward and waves side to side at shoulder height.",
            "emoji": "👋",
            "hand_shape": "Open palm",
            "movement": "Waving"
        },
        "thank you": {
            "description": "Flat hand from chin outward",
            "gesture_steps": [
                "Place flat hand at chin",
                "Move hand forward and down",
                "Palm faces down",
                "Single smooth motion"
            ],
            "audio_text": "In Auslan, thank you is signed by placing a flat hand at your chin and moving it forward and downward. This gesture represents sending your thanks to the other person.",
            "emoji": "🙏",
            "hand_shape": "Flat hand",
            "movement": "From chin outward"
        },
        "please": {
            "description": "Hand rubs chest in circle",
            "gesture_steps": [
                "Place flat hand on chest",
                "Rub in circular motion",
                "Clockwise circle",
                "Polite expression"
            ],
            "audio_text": "In Auslan, please is signed by rubbing your flat hand in a circular motion on your chest. This shows politeness when making a request.",
            "emoji": "🤲",
            "hand_shape": "Flat hand",
            "movement": "Circular on chest"
        },
        "sorry": {
            "description": "Fist circles on chest",
            "gesture_steps": [
                "Make a fist",
                "Place on chest",
                "Rub in circular motion",
                "Sorry expression"
            ],
            "audio_text": "In Auslan, sorry is signed by making a fist and rubbing it in a circular motion on your chest while showing an apologetic facial expression.",
            "emoji": "😔",
            "hand_shape": "Fist",
            "movement": "Circular on chest"
        },
        "help": {
            "description": "Thumbs up lifted by other hand",
            "gesture_steps": [
                "Make thumbs up sign",
                "Place on flat palm",
                "Lift both hands up",
                "Represents support"
            ],
            "audio_text": "In Auslan, help is signed by placing a thumbs up on your flat palm and lifting both hands upward. This symbolizes one person being supported by another.",
            "emoji": "🆘",
            "hand_shape": "Thumbs up on palm",
            "movement": "Lifting up"
        },
        "yes": {
            "description": "Fist nods like head",
            "gesture_steps": [
                "Make a fist",
                "Move up and down",
                "Like nodding",
                "Affirmative expression"
            ],
            "audio_text": "In Auslan, yes is signed by making a fist and moving it up and down like a nodding head. This clearly indicates agreement.",
            "emoji": "✅",
            "hand_shape": "Fist",
            "movement": "Nodding"
        },
        "no": {
            "description": "Finger wags side to side",
            "gesture_steps": [
                "Extend index finger",
                "Wag side to side",
                "Like saying no no",
                "Negative expression"
            ],
            "audio_text": "In Auslan, no is often signed by wagging your index finger side to side, similar to the common gesture meaning 'no no'.",
            "emoji": "❌",
            "hand_shape": "Index finger",
            "movement": "Wagging"
        },
        "love": {
            "description": "Arms crossed hugging chest",
            "gesture_steps": [
                "Cross arms over chest",
                "Hands on opposite shoulders",
                "Like hugging yourself",
                "Warm expression"
            ],
            "audio_text": "In Auslan, love is signed by crossing your arms over your chest in a self-hugging motion. This represents embracing someone you love.",
            "emoji": "❤️",
            "hand_shape": "Crossed arms",
            "movement": "Hugging"
        },
        "water": {
            "description": "W hand taps chin",
            "gesture_steps": [
                "Make W with three fingers",
                "Tap on chin",
                "Repeat 2-3 times",
                "W for water"
            ],
            "audio_text": "In Auslan, water is signed by forming a W shape with three fingers and tapping it on your chin. The W represents the first letter of water.",
            "emoji": "💧",
            "hand_shape": "W hand",
            "movement": "Tapping chin"
        },
        "food": {
            "description": "Fingers tap mouth",
            "gesture_steps": [
                "Bunch fingers together",
                "Tap on lips",
                "Repeat motion",
                "Mimics eating"
            ],
            "audio_text": "In Auslan, food is signed by bunching your fingers together and tapping them on your lips repeatedly, mimicking the action of eating.",
            "emoji": "🍽️",
            "hand_shape": "Bunched fingers",
            "movement": "Tapping lips"
        }
    },
    
    "LSF": {
        "name": "French Sign Language",
        "country": "France",
        "hello": {
            "description": "Hand waves near forehead",
            "gesture_steps": [
                "Raise hand near forehead",
                "Wave gently",
                "Palm facing outward",
                "Friendly greeting motion"
            ],
            "audio_text": "In French Sign Language, known as LSF, hello is signed by raising your hand near your forehead and waving gently with the palm facing outward. This is a warm, welcoming gesture.",
            "emoji": "👋",
            "hand_shape": "Open hand",
            "movement": "Waving near forehead"
        },
        "thank you": {
            "description": "Hand moves from mouth outward",
            "gesture_steps": [
                "Touch fingers to lips",
                "Move hand forward",
                "Palm turns upward",
                "Like blowing a thank you"
            ],
            "audio_text": "In French Sign Language, thank you is signed by touching your fingers to your lips and moving your hand forward with the palm turning upward. It's like sending a kiss of gratitude.",
            "emoji": "🙏",
            "hand_shape": "Flat hand",
            "movement": "From lips outward"
        },
        "please": {
            "description": "Hand on heart with slight bow",
            "gesture_steps": [
                "Place hand on heart",
                "Slight bow of head",
                "Sincere expression",
                "Shows humility"
            ],
            "audio_text": "In French Sign Language, please is signed by placing your hand on your heart with a slight bow of the head. This gesture shows sincerity and humility when making a request.",
            "emoji": "🤲",
            "hand_shape": "Hand on heart",
            "movement": "Slight bow"
        },
        "sorry": {
            "description": "Hand circles on chest",
            "gesture_steps": [
                "Place hand on chest",
                "Make circular motion",
                "Apologetic expression",
                "Shows remorse"
            ],
            "audio_text": "In French Sign Language, sorry is signed by placing your hand on your chest and making a circular motion while showing an apologetic expression.",
            "emoji": "😔",
            "hand_shape": "Flat hand",
            "movement": "Circular on chest"
        },
        "help": {
            "description": "One hand lifts the other",
            "gesture_steps": [
                "One hand flat palm up",
                "Other hand or fist on top",
                "Lift both upward",
                "Symbolizes support"
            ],
            "audio_text": "In French Sign Language, help is signed by placing one hand on top of another and lifting both upward together, representing the act of giving support.",
            "emoji": "🆘",
            "hand_shape": "Stacked hands",
            "movement": "Lifting"
        },
        "yes": {
            "description": "Head nods with fist",
            "gesture_steps": [
                "Make a fist",
                "Nod it up and down",
                "Like head nodding",
                "Clear agreement"
            ],
            "audio_text": "In French Sign Language, yes is signed by making a fist and nodding it up and down, mimicking a nodding head to show agreement.",
            "emoji": "✅",
            "hand_shape": "Fist",
            "movement": "Nodding"
        },
        "no": {
            "description": "Index finger wags",
            "gesture_steps": [
                "Extend index finger",
                "Wag side to side",
                "Palm facing out",
                "Clear negation"
            ],
            "audio_text": "In French Sign Language, no is signed by extending your index finger and wagging it side to side with the palm facing outward.",
            "emoji": "❌",
            "hand_shape": "Index finger",
            "movement": "Wagging"
        },
        "love": {
            "description": "Arms embrace chest",
            "gesture_steps": [
                "Cross arms over chest",
                "Hands on shoulders",
                "Hugging motion",
                "Loving expression"
            ],
            "audio_text": "In French Sign Language, love is signed by crossing your arms over your chest in an embracing motion, symbolizing affection.",
            "emoji": "❤️",
            "hand_shape": "Crossed arms",
            "movement": "Embracing"
        },
        "water": {
            "description": "Drinking motion",
            "gesture_steps": [
                "Cup your hand",
                "Bring to lips",
                "Tilt like drinking",
                "Natural gesture"
            ],
            "audio_text": "In French Sign Language, water is signed by cupping your hand and bringing it to your lips in a drinking motion.",
            "emoji": "💧",
            "hand_shape": "Cupped hand",
            "movement": "Drinking"
        },
        "food": {
            "description": "Fingers move to mouth",
            "gesture_steps": [
                "Bunch fingers together",
                "Move toward mouth",
                "Repeat motion",
                "Eating gesture"
            ],
            "audio_text": "In French Sign Language, food is signed by bunching your fingers together and moving them toward your mouth repeatedly, mimicking eating.",
            "emoji": "🍽️",
            "hand_shape": "Bunched fingers",
            "movement": "To mouth"
        }
    },
    
    "DGS": {
        "name": "German Sign Language",
        "country": "Germany",
        "hello": {
            "description": "Wave or salute gesture",
            "gesture_steps": [
                "Raise hand to head level",
                "Wave or give slight salute",
                "Friendly expression",
                "Clear greeting motion"
            ],
            "audio_text": "In German Sign Language, known as DGS, hello is signed with a wave or slight salute gesture at head level. This is a straightforward, friendly greeting.",
            "emoji": "👋",
            "hand_shape": "Open hand",
            "movement": "Wave or salute"
        },
        "thank you": {
            "description": "Hand from chin moves forward",
            "gesture_steps": [
                "Touch chin with fingertips",
                "Move hand forward and down",
                "Palm faces recipient",
                "Grateful expression"
            ],
            "audio_text": "In German Sign Language, thank you is signed by touching your chin with your fingertips and moving your hand forward and downward toward the person you're thanking.",
            "emoji": "🙏",
            "hand_shape": "Flat hand",
            "movement": "From chin forward"
        },
        "please": {
            "description": "Hand on chest, circular motion",
            "gesture_steps": [
                "Place hand on chest",
                "Small circular motion",
                "Polite expression",
                "Shows courtesy"
            ],
            "audio_text": "In German Sign Language, please is signed by placing your hand on your chest and making a small circular motion. This gesture shows courtesy and politeness.",
            "emoji": "🤲",
            "hand_shape": "Flat hand",
            "movement": "Circular on chest"
        },
        "sorry": {
            "description": "Fist rubs chest",
            "gesture_steps": [
                "Make a fist",
                "Place on chest",
                "Rub in circles",
                "Apologetic face"
            ],
            "audio_text": "In German Sign Language, sorry is signed by making a fist and rubbing it in circular motions on your chest while showing an apologetic expression.",
            "emoji": "😔",
            "hand_shape": "Fist",
            "movement": "Circular rubbing"
        },
        "help": {
            "description": "One hand supports and lifts other",
            "gesture_steps": [
                "Flat hand as base",
                "Fist or hand on top",
                "Lift upward together",
                "Supporting motion"
            ],
            "audio_text": "In German Sign Language, help is signed by placing one hand on top of another and lifting both upward, representing support and assistance.",
            "emoji": "🆘",
            "hand_shape": "Stacked hands",
            "movement": "Lifting"
        },
        "yes": {
            "description": "Fist nods up and down",
            "gesture_steps": [
                "Form a fist",
                "Move up and down",
                "Nodding motion",
                "Affirmative"
            ],
            "audio_text": "In German Sign Language, yes is signed by making a fist and moving it up and down in a nodding motion to indicate agreement.",
            "emoji": "✅",
            "hand_shape": "Fist",
            "movement": "Nodding"
        },
        "no": {
            "description": "Index finger waves side to side",
            "gesture_steps": [
                "Extend index finger",
                "Wave horizontally",
                "Clear side to side motion",
                "Negative expression"
            ],
            "audio_text": "In German Sign Language, no is signed by extending your index finger and waving it side to side in a clear horizontal motion.",
            "emoji": "❌",
            "hand_shape": "Index finger",
            "movement": "Side to side"
        },
        "love": {
            "description": "Arms cross over heart",
            "gesture_steps": [
                "Cross arms over chest",
                "Hands on shoulders",
                "Hugging yourself",
                "Warm expression"
            ],
            "audio_text": "In German Sign Language, love is signed by crossing your arms over your chest in a hugging motion, representing affection and care.",
            "emoji": "❤️",
            "hand_shape": "Crossed arms",
            "movement": "Self-hug"
        },
        "water": {
            "description": "W shape taps chin or drinking motion",
            "gesture_steps": [
                "Form W with fingers or cup hand",
                "Tap chin or mime drinking",
                "Clear water indication",
                "Simple gesture"
            ],
            "audio_text": "In German Sign Language, water can be signed by forming a W shape and tapping your chin, or by making a drinking motion with a cupped hand.",
            "emoji": "💧",
            "hand_shape": "W or cupped",
            "movement": "Tap or drink"
        },
        "food": {
            "description": "Hand moves to mouth",
            "gesture_steps": [
                "Bunch fingers together",
                "Bring to mouth",
                "Repeat eating motion",
                "Natural gesture"
            ],
            "audio_text": "In German Sign Language, food is signed by bunching your fingers together and bringing them to your mouth in a repeated eating motion.",
            "emoji": "🍽️",
            "hand_shape": "Bunched fingers",
            "movement": "To mouth"
        }
    },
    
    "CSL": {
        "name": "Chinese Sign Language",
        "country": "China",
        "hello": {
            "description": "Cupped hands move together, bow",
            "gesture_steps": [
                "Cup both hands together",
                "Hold in front of chest",
                "Slight bow",
                "Traditional respectful greeting"
            ],
            "audio_text": "In Chinese Sign Language, hello is often expressed by cupping both hands together in front of the chest and giving a slight bow. This reflects the traditional Chinese greeting of respect.",
            "emoji": "🙏",
            "hand_shape": "Cupped hands",
            "movement": "Together with bow"
        },
        "thank you": {
            "description": "Hands press together with bow",
            "gesture_steps": [
                "Press palms together",
                "Hold at chest level",
                "Bow from waist",
                "Shows deep gratitude"
            ],
            "audio_text": "In Chinese Sign Language, thank you is expressed by pressing your palms together at chest level and bowing. The depth of the bow shows the level of gratitude.",
            "emoji": "🙏",
            "hand_shape": "Pressed palms",
            "movement": "Bow"
        },
        "please": {
            "description": "Hands together with slight forward motion",
            "gesture_steps": [
                "Press hands together",
                "Move slightly forward",
                "Requesting posture",
                "Polite expression"
            ],
            "audio_text": "In Chinese Sign Language, please is signed with hands pressed together and moved slightly forward in a requesting gesture, showing politeness.",
            "emoji": "🤲",
            "hand_shape": "Hands together",
            "movement": "Forward"
        },
        "sorry": {
            "description": "Hand on heart with bow",
            "gesture_steps": [
                "Place hand on heart",
                "Bow deeply",
                "Show sincere regret",
                "May repeat bow"
            ],
            "audio_text": "In Chinese Sign Language, sorry is expressed by placing your hand on your heart and bowing deeply to show sincere regret and apology.",
            "emoji": "😔",
            "hand_shape": "Hand on heart",
            "movement": "Deep bow"
        },
        "help": {
            "description": "One hand supports other, lift",
            "gesture_steps": [
                "Base hand flat",
                "Other hand on top",
                "Lift together",
                "Supporting gesture"
            ],
            "audio_text": "In Chinese Sign Language, help is signed by placing one hand on top of another and lifting both upward, symbolizing giving support.",
            "emoji": "🆘",
            "hand_shape": "Layered hands",
            "movement": "Lifting"
        },
        "yes": {
            "description": "Nod with optional hand gesture",
            "gesture_steps": [
                "Nod head clearly",
                "May add hand movement",
                "Affirmative expression",
                "Clear agreement"
            ],
            "audio_text": "In Chinese Sign Language, yes is typically expressed through a clear head nod, often accompanied by an affirmative facial expression.",
            "emoji": "✅",
            "hand_shape": "Optional",
            "movement": "Nodding"
        },
        "no": {
            "description": "Hand waves in front",
            "gesture_steps": [
                "Open hand in front",
                "Wave side to side",
                "Or head shake",
                "Negative indication"
            ],
            "audio_text": "In Chinese Sign Language, no is often signed by waving an open hand in front of the body or shaking the head to indicate negation.",
            "emoji": "❌",
            "hand_shape": "Open hand",
            "movement": "Waving"
        },
        "love": {
            "description": "Hands form heart at chest",
            "gesture_steps": [
                "Form heart with hands",
                "Or cross arms on chest",
                "Warm expression",
                "Shows affection"
            ],
            "audio_text": "In Chinese Sign Language, love can be expressed by forming a heart shape with your hands or crossing your arms over your chest in a hugging motion.",
            "emoji": "❤️",
            "hand_shape": "Heart or crossed",
            "movement": "At chest"
        },
        "water": {
            "description": "Drinking motion with cupped hand",
            "gesture_steps": [
                "Cup your hand",
                "Bring to mouth",
                "Tipping motion",
                "Like drinking"
            ],
            "audio_text": "In Chinese Sign Language, water is signed by cupping your hand and bringing it to your mouth in a drinking motion.",
            "emoji": "💧",
            "hand_shape": "Cupped hand",
            "movement": "Drinking"
        },
        "food": {
            "description": "Chopstick motion to mouth",
            "gesture_steps": [
                "Hold fingers like chopsticks",
                "Move toward mouth",
                "Eating motion",
                "Culturally specific"
            ],
            "audio_text": "In Chinese Sign Language, food is often signed by mimicking the motion of using chopsticks to eat, bringing the hand toward the mouth.",
            "emoji": "🍽️",
            "hand_shape": "Chopstick hold",
            "movement": "To mouth"
        }
    },
    
    "KSL": {
        "name": "Korean Sign Language",
        "country": "South Korea",
        "hello": {
            "description": "Bow with hands at sides",
            "gesture_steps": [
                "Stand with hands at sides",
                "Bow from the waist",
                "Polite, respectful greeting",
                "Depth shows respect level"
            ],
            "audio_text": "In Korean Sign Language, hello is expressed through a bow with hands at your sides, reflecting Korean cultural practices. The depth of the bow indicates the level of respect.",
            "emoji": "🙇",
            "hand_shape": "Hands at sides",
            "movement": "Bowing"
        },
        "thank you": {
            "description": "Deep bow with hands together",
            "gesture_steps": [
                "Place hands together or at sides",
                "Bow deeply",
                "Hold briefly",
                "Grateful expression"
            ],
            "audio_text": "In Korean Sign Language, thank you is expressed through a deep bow, often with hands placed together or at the sides. This shows sincere gratitude in Korean culture.",
            "emoji": "🙏",
            "hand_shape": "Together or at sides",
            "movement": "Deep bow"
        },
        "please": {
            "description": "Slight bow with requesting gesture",
            "gesture_steps": [
                "Hands may be together",
                "Slight bow forward",
                "Polite expression",
                "Humble request"
            ],
            "audio_text": "In Korean Sign Language, please is signed with a slight bow and often hands placed together, showing humility and politeness when making a request.",
            "emoji": "🤲",
            "hand_shape": "Hands together",
            "movement": "Slight bow"
        },
        "sorry": {
            "description": "Deep bow showing apology",
            "gesture_steps": [
                "Stand properly",
                "Bow deeply",
                "Hold the bow",
                "Sincere regret shown"
            ],
            "audio_text": "In Korean Sign Language, sorry is expressed through a deep bow held for a moment to show sincere regret. This reflects the importance of apologies in Korean culture.",
            "emoji": "😔",
            "hand_shape": "At sides or together",
            "movement": "Deep bow"
        },
        "help": {
            "description": "One hand lifts another",
            "gesture_steps": [
                "Flat hand as base",
                "Fist on top",
                "Lift upward",
                "Support gesture"
            ],
            "audio_text": "In Korean Sign Language, help is signed by placing a fist on a flat palm and lifting both hands upward, representing giving support to someone.",
            "emoji": "🆘",
            "hand_shape": "Fist on palm",
            "movement": "Lifting"
        },
        "yes": {
            "description": "Nod with possible gesture",
            "gesture_steps": [
                "Nod head clearly",
                "May add hand motion",
                "Affirmative",
                "Clear agreement"
            ],
            "audio_text": "In Korean Sign Language, yes is expressed through a clear head nod, which may be accompanied by a hand gesture for emphasis.",
            "emoji": "✅",
            "hand_shape": "Optional",
            "movement": "Nodding"
        },
        "no": {
            "description": "Hand wave or head shake",
            "gesture_steps": [
                "Wave hand in front",
                "Or shake head",
                "Negative expression",
                "Clear refusal"
            ],
            "audio_text": "In Korean Sign Language, no is often expressed by waving the hand in front of the body or shaking the head to indicate negation.",
            "emoji": "❌",
            "hand_shape": "Open hand",
            "movement": "Waving or shaking"
        },
        "love": {
            "description": "Heart shape with hands",
            "gesture_steps": [
                "Form heart with fingers",
                "Hold at chest",
                "Or cross arms",
                "Loving expression"
            ],
            "audio_text": "In Korean Sign Language, love is often expressed by forming a heart shape with your fingers or hands, a gesture that has become very popular in Korean culture.",
            "emoji": "❤️",
            "hand_shape": "Heart shape",
            "movement": "At chest"
        },
        "water": {
            "description": "Drinking motion",
            "gesture_steps": [
                "Cup hand",
                "Bring to lips",
                "Drinking motion",
                "Clear gesture"
            ],
            "audio_text": "In Korean Sign Language, water is signed by cupping your hand and bringing it to your lips in a drinking motion.",
            "emoji": "💧",
            "hand_shape": "Cupped hand",
            "movement": "Drinking"
        },
        "food": {
            "description": "Chopstick or spoon motion",
            "gesture_steps": [
                "Mimic utensil holding",
                "Move to mouth",
                "Eating motion",
                "Culturally relevant"
            ],
            "audio_text": "In Korean Sign Language, food is signed by mimicking the motion of eating with chopsticks or a spoon, bringing the hand toward the mouth.",
            "emoji": "🍽️",
            "hand_shape": "Utensil hold",
            "movement": "To mouth"
        }
    }
}

# List of all available words across all languages
AVAILABLE_WORDS = ["hello", "thank you", "please", "sorry", "help", "yes", "no", "love", "water", "food"]

# List of all supported languages
SUPPORTED_LANGUAGES = list(SIGN_LANGUAGE_DATA.keys())

def get_sign_data(language: str, word: str) -> dict:
    """
    Get sign language data for a specific language and word.
    Returns None if not found.
    """
    lang_upper = language.upper()
    word_lower = word.lower()
    
    if lang_upper not in SIGN_LANGUAGE_DATA:
        return None
    
    lang_data = SIGN_LANGUAGE_DATA[lang_upper]
    
    if word_lower not in lang_data:
        return None
    
    word_data = lang_data[word_lower]
    return {
        "language": lang_upper,
        "language_name": lang_data["name"],
        "country": lang_data["country"],
        "word": word_lower,
        **word_data
    }

def get_all_languages_for_word(word: str) -> list:
    """
    Get sign data for a word across ALL supported languages.
    Returns a list of sign data for each language.
    """
    word_lower = word.lower()
    results = []
    
    for lang_code, lang_data in SIGN_LANGUAGE_DATA.items():
        if word_lower in lang_data:
            word_data = lang_data[word_lower]
            results.append({
                "language": lang_code,
                "language_name": lang_data["name"],
                "country": lang_data["country"],
                "word": word_lower,
                **word_data
            })
    
    return results

def get_language_info(language: str) -> dict:
    """
    Get basic information about a sign language.
    """
    lang_upper = language.upper()
    
    if lang_upper not in SIGN_LANGUAGE_DATA:
        return None
    
    lang_data = SIGN_LANGUAGE_DATA[lang_upper]
    return {
        "code": lang_upper,
        "name": lang_data["name"],
        "country": lang_data["country"],
        "available_words": [w for w in AVAILABLE_WORDS if w in lang_data]
    }
