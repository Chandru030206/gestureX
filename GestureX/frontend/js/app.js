const API_BASE = 'http://localhost:8000';

let currentLanguage = 'ASL';
let currentMode = null;
let isRecording = false;
let video = null;
let canvas = null;
let ctx = null;
let predictionInterval = null;
let sentenceWords = [];
let lastGesture = null;
let gestureHoldTime = 0;
// Sentence builder timing: require 1.0s stability before committing
const HOLD_THRESHOLD = 1000;

// =====================================================
// Recognition policy (word mode vs alphabet mode)
// =====================================================
// Only accept predictions above this confidence.
const WORD_CONFIDENCE_THRESHOLD = 0.7;
// BLANK is allowed only if it's *very* confident (usually indicates no hand).
const BLANK_CONFIDENCE_THRESHOLD = 0.95;
// Temporal smoothing: require the same prediction for N consecutive frames.
const STABLE_FRAMES_REQUIRED = 5;

// Alphabet gating: ignore low-confidence letters (reduces random letters when hand is unclear)
const LETTER_CONFIDENCE_THRESHOLD = 0.85;
// Require slightly fewer stable frames for letters than words (letters take longer to hold)
const STABLE_LETTER_FRAMES_REQUIRED = 4;

// "No hand detected" grace period
const NO_HAND_TIMEOUT_MS = 2000;
let lastHandDetectedAt = Date.now();

function isBlankToken(token) {
    if (typeof token !== 'string') return false;
    return token.trim().toUpperCase() === 'BLANK';
}

const MODE_WORD = 'WORD';
const MODE_ALPHABET = 'ALPHABET';

let recognitionMode = MODE_WORD;

// Word smoothing state (frontend-side)
let stableWordLabel = null;
let stableWordCount = 0;
let lastCommittedWord = null;
let lastCommittedWordAt = 0;
const WORD_COMMIT_COOLDOWN_MS = 1200; // prevent duplicate commits when UI holds steady

// Alphabet buffer (frontend-side) with its own smoothing
let alphabetBuffer = [];
let stableLetterLabel = null;
let stableLetterCount = 0;
let lastCommittedLetter = null;
let lastCommittedLetterAt = 0;
const LETTER_COMMIT_COOLDOWN_MS = 700;

function setRecognitionMode(mode) {
    recognitionMode = mode;
}

function resetWordSmoothing() {
    stableWordLabel = null;
    stableWordCount = 0;
}

function resetLetterSmoothing() {
    stableLetterLabel = null;
    stableLetterCount = 0;
}

function resetAlphabetBuffer() {
    alphabetBuffer = [];
    // Keep existing UI helpers consistent with the rest of the file
    detectedLetters = [];
    updateDetectedLettersUI();
    updateDetectedNameUI();
}

// Capture/prediction pacing
// Slower intervals reduce flicker and make results feel more stable.
const GESTURE_PREDICT_INTERVAL_MS = 900;
const FINGERSPELL_PREDICT_INTERVAL_MS = 600;

const LANGUAGES = {
    ASL: 'American Sign Language',
    BSL: 'British Sign Language',
    ISL: 'Indian Sign Language',
    JSL: 'Japanese Sign Language',
    AUSLAN: 'Australian Sign Language',
    LSF: 'French Sign Language',
    DGS: 'German Sign Language',
    LIBRAS: 'Brazilian Sign Language',
    KSL: 'Korean Sign Language',
    CSL: 'Chinese Sign Language'
};

document.addEventListener('DOMContentLoaded', function() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    if (canvas) {
        ctx = canvas.getContext('2d');
        // Increased resolution for better gesture recognition and a larger camera view
        canvas.width = 960;
        canvas.height = 720;
    }
    sentenceWords = [];
    renderSentence();
    checkBackendHealth();
    // Prepare fingerspell elements early so start button logic can access them
    try { initFingerspellElements(); } catch (e) { console.warn('Fingerspell elements init skipped:', e); }
    
    var textInput = document.getElementById('text-input');
    if (textInput) {
        textInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') convertToGesture();
        });
    }
});

async function checkBackendHealth() {
    var statusEl = document.getElementById('connection-status');
    try {
        var res = await Promise.resolve({ok:true, json:()=>Promise.resolve({success:false})});
        if (res.ok) {
            statusEl.textContent = 'Backend: Connected';
            statusEl.style.color = '#00ff88';
        } else {
            throw new Error('fail');
        }
    } catch (e) {
        statusEl.textContent = 'Backend: Disconnected';
        statusEl.style.color = '#ff6b6b';
    }
}

function selectLanguage(lang) {
    currentLanguage = lang;
    document.querySelectorAll('.lang-btn').forEach(function(btn) {
        btn.classList.toggle('active', btn.dataset.lang === lang);
    });
    document.getElementById('current-language').textContent = lang + ' - ' + LANGUAGES[lang];
    document.getElementById('gesture-lang-badge').textContent = lang;
    document.getElementById('speech-lang-badge').textContent = lang;
    
    Promise.resolve({ok:true, json:()=>Promise.resolve({success:false})});
}

function selectMode(mode) {
    currentMode = mode;
    document.getElementById('mode-selection').classList.add('hidden');
    document.getElementById('language-section').classList.add('hidden');
    
    if (mode === 'gesture_to_speech') {
        document.getElementById('gesture-to-speech').classList.remove('hidden');
    } else {
        document.getElementById('speech-to-gesture').classList.remove('hidden');
    }
    
    Promise.resolve({ok:true, json:()=>Promise.resolve({success:false})});
}

function goBack() {
    stopCamera();
    document.getElementById('mode-selection').classList.remove('hidden');
    document.getElementById('language-section').classList.remove('hidden');
    document.getElementById('gesture-to-speech').classList.add('hidden');
    document.getElementById('speech-to-gesture').classList.add('hidden');
    currentMode = null;
}

async function startCamera() {
    try {
        var stream = await navigator.mediaDevices.getUserMedia({
            // Request a larger video stream for clearer input (keeps aspect ratio 4:3)
            video: { facingMode: 'user', width: 960, height: 720 }
        });
        video.srcObject = stream;
        await video.play();
        
        document.getElementById('camera-overlay').classList.add('hidden');
        document.getElementById('start-camera').classList.add('hidden');
        document.getElementById('stop-camera').classList.remove('hidden');
        
        isRecording = true;
    predictionInterval = setInterval(captureAndPredict, GESTURE_PREDICT_INTERVAL_MS);
        console.log('Camera started');
        // Also start the fingerspell session and detector in the background so a single
        // camera feeds both predictors. Use currentLanguage as the fingerspell language.
        try {
            // Ensure fingerspell elements point to the same video/canvas
            fingerspellVideo = video;
            fingerspellCanvas = canvas;
            fingerspellCtx = ctx;

            // Force ASL alphabet for fingerspelling detection (preferred for name spelling)
            const sessRes = await Promise.resolve({ok:true, json:()=>Promise.resolve({success:false})});
            const sessJson = await sessRes.json();
            if (sessJson && sessJson.session_id) {
                fingerspellSessionId = sessJson.session_id;
                isFingerSpelling = true;
                // start fingerspell loop if not already started
                if (!fingerspellInterval) {
                    fingerspellInterval = setInterval(captureAndDetectLetter, FINGERSPELL_PREDICT_INTERVAL_MS);
                }
                // reset debug counters
                fingerspellFramesSent = 0;
                fingerspellLastResponse = null;
                updateFingerspellDebug();
            }
        } catch (e) {
            console.warn('Could not start fingerspell session:', e);
        }
    } catch (err) {
        console.error('Camera error:', err);
        alert('Could not access camera');
    }
}

function stopCamera() {
    isRecording = false;
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(function(t) { t.stop(); });
        video.srcObject = null;
    }
    if (predictionInterval) {
        clearInterval(predictionInterval);
        predictionInterval = null;
    }
    // Stop fingerspell detector if running and notify backend to finalize
    if (fingerspellInterval) {
        clearInterval(fingerspellInterval);
        fingerspellInterval = null;
    }
    if (isFingerSpelling && fingerspellSessionId) {
        Promise.resolve({ok:true, json:()=>Promise.resolve({success:false})})
        .then(res => res.json()).then(js => {
            if (js && js.detected_name) {
                addToSentence(js.detected_name);
            }
        }).catch(err => console.warn('Fingerspell stop error', err));
    }
    
    document.getElementById('camera-overlay').classList.remove('hidden');
    document.getElementById('start-camera').classList.remove('hidden');
    document.getElementById('stop-camera').classList.add('hidden');
    
    document.getElementById('gesture-emoji').textContent = '';
    document.getElementById('gesture-name').textContent = 'None';
    document.getElementById('confidence-fill').style.width = '0%';
    document.getElementById('confidence-text').textContent = '0%';
    document.getElementById('speech-text').textContent = '-';
    document.getElementById('speak-btn').disabled = true;
    
    lastGesture = null;
    gestureHoldTime = 0;
}

async function captureAndPredict() {
    if (!isRecording || !video || !ctx) return;
    
    try {
        // Draw directly without flipping - backend will handle it
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        var imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        var res = await Promise.resolve({ok:true, json:()=>Promise.resolve({success:false})})
        });
        
        var result = null;
        try { result = await res.json(); } catch (e) { result = { error: 'Invalid JSON', status: res.status }; }
        console.log('Prediction:', result);
        gestureLastResponse = result;
        updateGestureUI(result);
        // update global debug panel
        updateGlobalDebug();
    } catch (err) {
        console.error('Prediction error:', err);
    }
}

// Buffers for Duo Logic
let duoLetterBuffer = [];
let duoLastLetterAt = 0;
let duoLetterTimer = null;
const FINGERSPELL_PAUSE_THRESHOLD = 1500; // 1.5s

function updateGestureUI(result) {
    const emojiEl = document.getElementById('gesture-emoji');
    const nameEl = document.getElementById('gesture-name');
    const confFill = document.getElementById('confidence-fill');
    const confText = document.getElementById('confidence-text');
    const speechEl = document.getElementById('speech-text');
    const speakBtn = document.getElementById('speak-btn');
    
    if (!result || !result.success) {
        if (result && result.display_text === "No hand detected") {
            nameEl.textContent = "No hand detected";
            nameEl.style.color = "#ff4d4d";
        }
        return;
    }

    const { text, confidence, type } = result;
    const confPct = (confidence * 100).toFixed(1);

    // Update common UI elements
    confFill.style.width = confPct + '%';
    confText.textContent = confPct + '%';
    
    // Handle "Uncertain"
    if (text === "Uncertain") {
        nameEl.textContent = "Uncertain";
        nameEl.style.color = "#ffcc00";
        return;
    }

    nameEl.style.color = "#00ff88";

    if (type === "word") {
        // Clear alphabet buffer if we detect a word
        finalizeDuoLetters();
        
        nameEl.textContent = text;
        speechEl.textContent = text;
        speakBtn.disabled = false;
        
        // Add to sentence (backend already does 5-frame smoothing)
        addToSentence(text);
        
    } else if (type === "alphabet") {
        nameEl.textContent = "Letter: " + text;
        
        // Add to buffer if it's a new letter
        if (duoLetterBuffer[duoLetterBuffer.length - 1] !== text) {
            duoLetterBuffer.push(text);
            updateDuoLettersUI();
        }
        
        // ALWAYS reset timer on EACH detection of the letter (including holds)
        duoLastLetterAt = Date.now();
        if (duoLetterTimer) clearTimeout(duoLetterTimer);
        duoLetterTimer = setTimeout(finalizeDuoLetters, FINGERSPELL_PAUSE_THRESHOLD);
    }
}

function updateDuoLettersUI() {
    const letterGrid = document.getElementById('micro-detected-letters-grid');
    const currentLetter = document.getElementById('micro-current-detected-letter');
    
    if (duoLetterBuffer.length > 0) {
        currentLetter.textContent = duoLetterBuffer[duoLetterBuffer.length - 1];
        letterGrid.textContent = duoLetterBuffer.join(" ");
    } else {
        currentLetter.textContent = "-";
        letterGrid.textContent = "Letters will appear here...";
    }
}

function finalizeDuoLetters() {
    if (duoLetterBuffer.length === 0) return;
    
    const word = duoLetterBuffer.join("");
    if (word.length > 0) {
        addToSentence(`Your name is ${word}`);
    }
    
    duoLetterBuffer = [];
    updateDuoLettersUI();
    if (duoLetterTimer) clearTimeout(duoLetterTimer);
}

function speakText() {
    var text = document.getElementById('speech-text').textContent;
    if (text && text !== '-') speak(text);
}

function speak(text) {
    if (!text) return;
    speechSynthesis.cancel();
    var utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    speechSynthesis.speak(utterance);
}

function addToSentence(word) {
    if (!word || word === lastGesture) return;
    
    // Simple frontend smoothing to avoid duplicates in the same second
    const now = Date.now();
    if (word === lastGesture && now - lastCommittedWordAt < 1000) return;
    
    sentenceWords.push(word);
    lastGesture = word;
    lastCommittedWordAt = now;
    
    renderSentence();
    showFeedback('+ ' + word);
    speak(word);
}

function showFeedback(text) {
    var el = document.createElement('div');
    el.textContent = text;
    el.style.cssText = 'position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:#0f1724;color:#e6eef8;padding:14px 28px;border-radius:10px;font-size:1.1rem;font-weight:700;z-index:10000;animation:popFade 1s forwards;box-shadow:0 8px 20px rgba(2,6,23,0.6);';
    document.body.appendChild(el);
    setTimeout(function() { el.remove(); }, 1000);
}

function renderSentence() {
    var container = document.getElementById('sentence-words');
    if (!container) return;
    if (sentenceWords.length === 0) {
        container.innerHTML = '<span style="color:#666;font-style:italic">Hold gesture 1.5s to add words...</span>';
        return;
    }
    var html = '';
    for (var i = 0; i < sentenceWords.length; i++) {
        html += '<span class="word-chip">' + sentenceWords[i] + ' <button class="chip-remove" onclick="removeWord(' + i + ')">x</button></span> ';
    }
    container.innerHTML = html;
}

function removeWord(index) {
    sentenceWords.splice(index, 1);
    renderSentence();
}

function clearSentence() {
    sentenceWords = [];
    renderSentence();
}

function speakSentence() {
    if (sentenceWords.length > 0) {
        speak(sentenceWords.join(' '));
    }
}

function setPhrase(phrase) {
    document.getElementById('text-input').value = phrase;
    convertToGesture();
}

async function convertToGesture() {
    var input = document.getElementById('text-input');
    var resultsEl = document.getElementById('gesture-results');
    var text = input.value.trim();
    
    if (!text) {
        resultsEl.innerHTML = '<div class="placeholder"><p>Enter text to see gestures</p></div>';
        return;
    }
    
    resultsEl.innerHTML = '<div class="placeholder"><p>Finding gestures...</p></div>';
    
    try {
        var res = await Promise.resolve({ok:true, json:()=>Promise.resolve({success:false})});
        var result = await res.json();
        displayGestureResults(result, text);
    } catch (err) {
        resultsEl.innerHTML = '<div class="placeholder"><p>Server error</p></div>';
    }
}

function displayGestureResults(result, originalText) {
    var resultsEl = document.getElementById('gesture-results');
    
    if (!result.success) {
        resultsEl.innerHTML = '<div class="placeholder"><p>' + (result.error || 'Not found') + '</p></div>';
        return;
    }
    
    var html = '<h3 class="sequence-title">Sign Sequence</h3><div class="gesture-sequence">';
    var gestures = result.gestures || [result];

    // Fallback: if backend didn't return gestures and the original text looks like letters/word(s),
    // show a fingerspelling fallback so users can see all alphabet letters (useful for short words like "are").
    var allLettersOnly = /^[\sA-Za-z]+$/.test(originalText);
    if ((Array.isArray(gestures) && gestures.length === 0) ||
        (gestures.length === 1 && !gestures[0].type && !gestures[0].word && allLettersOnly)) {
        var cleaned = originalText.replace(/[^A-Za-z]/g, '');
        if (cleaned.length > 0) {
            var lettersArr = cleaned.split('').map(function(ch) { return { letter: ch.toUpperCase() }; });
            gestures = [{ type: 'fingerspelling', word: originalText.trim(), letters: lettersArr }];
        }
    }
    
    for (var i = 0; i < gestures.length; i++) {
        var g = gestures[i];
        var isActive = i === 2; // Highlight the 3rd card (current focus)
        var activeClass = isActive ? ' active' : '';
        
        if (g.type === 'fingerspelling' && g.letters) {
            // Fingerspelling card with individual letters
            var letters = '';
            for (var j = 0; j < g.letters.length; j++) {
                letters += '<span class="letter-badge">' + g.letters[j].letter + '</span>';
            }
            html += '<div class="gesture-card fingerspell' + activeClass + '">' +
                '<div class="card-num">' + (i+1) + '</div>' +
                '<div class="card-word">' + (g.word || '').charAt(0).toUpperCase() + (g.word || '').slice(1) + '</div>' +
                '<div class="card-icon">' + getGestureIcon('fingerspell') + '</div>' +
                '<div class="fingerspell-letters">' + letters + '</div>' +
                '<div class="card-hint">Fingerspell each letter</div>' +
                '</div>';
        } else {
            // Regular gesture card - show gesture icon
            var word = (g.word || '').toLowerCase();
            html += '<div class="gesture-card' + activeClass + '">' +
                '<div class="card-num">' + (i+1) + '</div>' +
                '<div class="card-word">' + (g.word || '').charAt(0).toUpperCase() + (g.word || '').slice(1) + '</div>' +
                '<div class="card-icon">' + getGestureIcon(word) + '</div>' +
                '<div class="card-desc">' + (g.description || '') + '</div>' +
                '<div class="card-hint">' + (g.hand_shape || '') + '</div>' +
                '</div>';
        }
    }
    
    html += '</div><div class="audio-control"><button class="btn btn-audio" onclick="playResultAudio(\'' + originalText.replace(/'/g, '') + '\')">Play Audio</button></div>';
    resultsEl.innerHTML = html;
}

// Hand gesture SVG icons for different signs
function getGestureIcon(word) {
    var icons = {
        // Common words with hand gesture representations
        'hello': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 8c-2 0-4 1-4 4v20h-4V16c0-3-2-4-4-4s-4 1-4 4v24l-4-4c-2-2-5-2-7 0s-2 5 0 7l12 14c2 2 5 4 8 4h10c6 0 10-4 10-10V20c0-3-2-4-4-4s-4 1-4 4v12h-4V12c0-3-2-4-4-4z" fill="#000000"/></svg>',
        'hi': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 8c-2 0-4 1-4 4v20h-4V16c0-3-2-4-4-4s-4 1-4 4v24l-4-4c-2-2-5-2-7 0s-2 5 0 7l12 14c2 2 5 4 8 4h10c6 0 10-4 10-10V20c0-3-2-4-4-4s-4 1-4 4v12h-4V12c0-3-2-4-4-4z" fill="#000000"/></svg>',
        'goodbye': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 8c-2 0-4 1-4 4v20h-4V16c0-3-2-4-4-4s-4 1-4 4v24l-4-4c-2-2-5-2-7 0s-2 5 0 7l12 14c2 2 5 4 8 4h10c6 0 10-4 10-10V20c0-3-2-4-4-4s-4 1-4 4v12h-4V12c0-3-2-4-4-4z" fill="#000000"/><path d="M44 12l8 8M52 12l-8 8" stroke="#000000" stroke-width="2"/></svg>',
        'yes': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M24 20c0-8 6-12 12-12s8 4 8 8v28c0 4-4 8-8 8H24c-4 0-8-4-8-8V28c0-4 2-8 8-8z" fill="#000000"/><path d="M32 24v8M32 40v-4" stroke="#fff" stroke-width="3" stroke-linecap="round"/></svg>',
        'no': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 24c0-4 2-8 6-8h12c4 0 8 4 8 8v16c0 4-4 8-8 8H26c-4 0-6-4-6-8V24z" fill="#000000"/><path d="M28 28l8 8M36 28l-8 8" stroke="#fff" stroke-width="2"/></svg>',
        'thank': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 52c-12 0-20-8-20-20V20c0-4 2-8 6-8h28c4 0 6 4 6 8v12c0 12-8 20-20 20z" fill="#000000"/><path d="M24 28c0-2 2-4 4-4s4 2 4 4M32 28c0-2 2-4 4-4s4 2 4 4M32 36v4M28 44h8" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'thank you': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 52c-12 0-20-8-20-20V20c0-4 2-8 6-8h28c4 0 6 4 6 8v12c0 12-8 20-20 20z" fill="#000000"/><path d="M24 28c0-2 2-4 4-4s4 2 4 4M32 28c0-2 2-4 4-4s4 2 4 4M32 36v4M28 44h8" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'thanks': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 52c-12 0-20-8-20-20V20c0-4 2-8 6-8h28c4 0 6 4 6 8v12c0 12-8 20-20 20z" fill="#000000"/><path d="M24 28c0-2 2-4 4-4s4 2 4 4M32 28c0-2 2-4 4-4s4 2 4 4M32 36v4M28 44h8" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'please': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="20" fill="#000000"/><path d="M32 24v8M28 36c0 2 2 4 4 4s4-2 4-4" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'sorry': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="20" fill="#000000"/><path d="M24 28a2 2 0 104 0 2 2 0 00-4 0M36 28a2 2 0 104 0 2 2 0 00-4 0M26 40c2-2 4-3 6-3s4 1 6 3" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'help': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 36l-4 8h8l-4-8zM32 20v24M32 20l12-8v16l-12 8-12-8V12l12 8z" fill="#000000"/><path d="M28 32h8M32 28v8" stroke="#fff" stroke-width="2"/></svg>',
        'stop': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 20h32v28c0 4-4 4-4 4H20s-4 0-4-4V20z" fill="#000000"/><path d="M20 12h4v8h-4zM28 8h4v12h-4zM36 8h4v12h-4zM44 12h4v8h-4z" fill="#000000"/></svg>',
        'i': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M28 20h8v28h-8z" fill="#000000"/><circle cx="32" cy="12" r="4" fill="#000000"/></svg>',
        'me': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M28 20h8v28h-8z" fill="#000000"/><circle cx="32" cy="12" r="4" fill="#000000"/><path d="M32 32l-8 8" stroke="#000000" stroke-width="3"/></svg>',
        'my': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 24c0-4 4-8 8-8h8c4 0 8 4 8 8v20c0 4-4 8-8 8h-8c-4 0-8-4-8-8V24z" fill="#000000"/><path d="M32 28v12" stroke="#fff" stroke-width="3" stroke-linecap="round"/></svg>',
        'you': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 16l16 12-16 12V16z" fill="#000000"/><path d="M16 28h24" stroke="#000000" stroke-width="4" stroke-linecap="round"/></svg>',
        'your': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 16l16 12-16 12V16z" fill="#000000"/><path d="M16 28h24" stroke="#000000" stroke-width="4" stroke-linecap="round"/></svg>',
        'we': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="24" cy="28" r="8" fill="#000000"/><circle cx="40" cy="28" r="8" fill="#000000"/><path d="M20 40h24v8H20z" fill="#000000"/></svg>',
        'love': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 52L16 36c-6-6-6-16 0-22s16-6 22 0l-6 6 6-6c6-6 16-6 22 0s6 16 0 22L32 52z" fill="#000000"/></svg>',
        'good': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 8c-2 0-4 2-4 4v20c0 2-2 4-4 4h-8c-2 0-4 2-4 4v8c0 2 2 4 4 4h24c6 0 10-4 10-10V24c0-2-2-4-4-4h-4c-2 0-4-2-4-4v-4c0-2-2-4-4-4h-2z" fill="#000000"/></svg>',
        'bad': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 56c-2 0-4-2-4-4V32c0-2-2-4-4-4h-8c-2 0-4-2-4-4v-8c0-2 2-4 4-4h24c6 0 10 4 10 10v16c0 2-2 4-4 4h-4c-2 0-4 2-4 4v4c0 2-2 4-4 4h-2z" fill="#000000"/></svg>',
        'what': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="20" stroke="#000000" stroke-width="4" fill="none"/><path d="M28 24c0-4 4-6 8-4s4 6 0 8l-4 4v4" stroke="#000000" stroke-width="3" stroke-linecap="round"/><circle cx="32" cy="48" r="2" fill="#000000"/></svg>',
        'where': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 8c-10 0-18 8-18 18 0 14 18 30 18 30s18-16 18-30c0-10-8-18-18-18z" fill="#000000"/><circle cx="32" cy="26" r="6" fill="#fff"/></svg>',
        'when': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="20" stroke="#000000" stroke-width="4" fill="none"/><path d="M32 20v14l8 8" stroke="#000000" stroke-width="3" stroke-linecap="round"/></svg>',
        'why': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M24 16l8 16 8-16M32 32v12" stroke="#000000" stroke-width="4" stroke-linecap="round"/><circle cx="32" cy="52" r="3" fill="#000000"/></svg>',
        'how': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 32c0-8 8-16 16-16s16 8 16 16-8 16-16 16" stroke="#000000" stroke-width="4" fill="none"/><path d="M32 48l-6-6h12l-6 6z" fill="#000000"/></svg>',
        'who': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="20" r="10" fill="#000000"/><path d="M18 52c0-8 6-14 14-14s14 6 14 14" stroke="#000000" stroke-width="4" fill="none"/></svg>',
        'name': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="12" y="16" width="40" height="32" rx="4" fill="#000000"/><path d="M20 28h24M20 36h16" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'is': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="16" fill="#000000"/><path d="M26 32h12" stroke="#fff" stroke-width="3" stroke-linecap="round"/></svg>',
        'are': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="24" cy="32" r="10" fill="#000000"/><circle cx="40" cy="32" r="10" fill="#000000"/></svg>',
        'am': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="16" fill="#000000"/><circle cx="32" cy="32" r="6" fill="#fff"/></svg>',
        'want': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 32h24M40 24l8 8-8 8" stroke="#000000" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        'need': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 12v40M20 24l12-12 12 12" stroke="#000000" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        'have': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 28c0-6 6-12 12-12s12 6 12 12v16c0 2-2 4-4 4H24c-2 0-4-2-4-4V28z" fill="#000000"/><path d="M28 32h8" stroke="#fff" stroke-width="2"/></svg>',
        'like': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 48L20 36c-4-4-4-10 0-14s10-4 14 0l-2 2 2-2c4-4 10-4 14 0s4 10 0 14L32 48z" fill="#000000"/></svg>',
        'know': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="24" r="12" fill="#000000"/><path d="M32 40v8M28 52h8" stroke="#000000" stroke-width="3" stroke-linecap="round"/></svg>',
        'think': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="28" r="16" fill="#000000"/><circle cx="24" cy="52" r="4" fill="#000000"/><circle cx="16" cy="56" r="2" fill="#000000"/></svg>',
        'feel': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 52L16 36c-6-6-6-14 0-20s14-6 20 0l-4 4 4-4c6-6 14-6 20 0s6 14 0 20L32 52z" stroke="#000000" stroke-width="3" fill="none"/></svg>',
        'see': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><ellipse cx="32" cy="32" rx="20" ry="12" stroke="#000000" stroke-width="3" fill="none"/><circle cx="32" cy="32" r="6" fill="#000000"/></svg>',
        'hear': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M24 24c0-6 4-10 10-10s10 4 10 10v16c0 6-4 10-10 10s-10-4-10-10" stroke="#000000" stroke-width="3" fill="none"/><path d="M16 28v8M48 28v8" stroke="#000000" stroke-width="3" stroke-linecap="round"/></svg>',
        'eat': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="36" r="16" fill="#000000"/><path d="M24 28l8 8 8-8" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'drink': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 16h24l-4 36H24L20 16z" fill="#000000"/><path d="M28 24v16" stroke="#fff" stroke-width="2"/></svg>',
        'water': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 8L20 28c-4 8 0 20 12 20s16-12 12-20L32 8z" fill="#000000"/></svg>',
        'food': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="18" fill="#000000"/><path d="M24 28h16M24 36h16" stroke="#fff" stroke-width="2"/></svg>',
        'home': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 32l20-20 20 20v20H12V32z" fill="#000000"/><rect x="26" y="36" width="12" height="16" fill="#fff"/></svg>',
        'work': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="16" y="20" width="32" height="28" rx="2" fill="#000000"/><path d="M24 20v-6h16v6" stroke="#000000" stroke-width="3"/><path d="M24 32h16M24 40h8" stroke="#fff" stroke-width="2"/></svg>',
        'school': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M32 8l24 12v8H8v-8L32 8z" fill="#000000"/><rect x="16" y="28" width="32" height="24" fill="#000000"/><rect x="28" y="36" width="8" height="16" fill="#fff"/></svg>',
        'friend': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="24" cy="20" r="8" fill="#000000"/><circle cx="40" cy="20" r="8" fill="#000000"/><path d="M12 48c0-8 6-14 12-14h16c6 0 12 6 12 14" stroke="#000000" stroke-width="3" fill="none"/></svg>',
        'family': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="16" r="6" fill="#000000"/><circle cx="20" cy="28" r="5" fill="#000000"/><circle cx="44" cy="28" r="5" fill="#000000"/><path d="M16 52c0-6 4-10 8-10h16c4 0 8 4 8 10" stroke="#000000" stroke-width="3" fill="none"/></svg>',
        'mother': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="20" r="10" fill="#000000"/><path d="M20 52c0-8 6-14 12-14s12 6 12 14" stroke="#000000" stroke-width="3" fill="none"/><path d="M28 16c0-4 2-8 4-8s4 4 4 8" stroke="#000000" stroke-width="2"/></svg>',
        'father': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="20" r="10" fill="#000000"/><path d="M20 52c0-8 6-14 12-14s12 6 12 14" stroke="#000000" stroke-width="3" fill="none"/><rect x="28" y="8" width="8" height="4" fill="#000000"/></svg>',
        'man': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="18" r="10" fill="#000000"/><path d="M32 28v20M20 36h24M32 48l-8 8M32 48l8 8" stroke="#000000" stroke-width="3" stroke-linecap="round"/></svg>',
        'woman': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="18" r="10" fill="#000000"/><path d="M24 28l8 28 8-28" fill="#000000"/><path d="M20 40h24" stroke="#000000" stroke-width="3"/></svg>',
        'boy': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="20" r="8" fill="#000000"/><path d="M32 28v16M24 34h16M32 44l-6 8M32 44l6 8" stroke="#000000" stroke-width="2" stroke-linecap="round"/></svg>',
        'girl': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="20" r="8" fill="#000000"/><path d="M26 28l6 24 6-24" fill="#000000"/><path d="M22 38h20" stroke="#000000" stroke-width="2"/></svg>',
        'today': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="12" y="16" width="40" height="36" rx="4" fill="#000000"/><path d="M12 28h40" stroke="#fff" stroke-width="2"/><circle cx="32" cy="40" r="6" fill="#fff"/></svg>',
        'tomorrow': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="12" y="16" width="40" height="36" rx="4" fill="#000000"/><path d="M12 28h40M36 40l8-4-8-4" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'yesterday': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="12" y="16" width="40" height="36" rx="4" fill="#000000"/><path d="M12 28h40M28 40l-8-4 8-4" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'time': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="20" stroke="#000000" stroke-width="4" fill="none"/><path d="M32 20v14l10 6" stroke="#000000" stroke-width="3" stroke-linecap="round"/></svg>',
        'money': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="32" cy="32" r="18" fill="#000000"/><path d="M32 20v24M26 26h8c2 0 4 2 4 4s-2 4-4 4h-4c-2 0-4 2-4 4s2 4 4 4h8" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'car': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M8 36h48v12H8z" fill="#000000"/><path d="M16 36l4-12h24l4 12" fill="#000000"/><circle cx="18" cy="48" r="4" fill="#000000"/><circle cx="46" cy="48" r="4" fill="#000000"/></svg>',
        'go': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 32h24M40 24l8 8-8 8" stroke="#000000" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        'come': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M48 32H24M24 24l-8 8 8 8" stroke="#000000" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>',
        'wait': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M24 16v32M40 16v32M24 32h16" stroke="#000000" stroke-width="4" stroke-linecap="round"/></svg>',
        'learn': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 20h40v32H12z" fill="#000000"/><path d="M20 28h24M20 36h16M20 44h20" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>',
        'teach': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="8" y="16" width="32" height="24" fill="#000000"/><circle cx="48" cy="24" r="8" fill="#000000"/><path d="M48 32v16" stroke="#000000" stroke-width="3"/></svg>',
        'fingerspell': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 24v24h4V24h-4zM24 20v28h4V20h-4zM32 16v32h4V16h-4zM40 20v28h4V20h-4zM48 24v24h4V24h-4z" fill="#000000"/></svg>',
        'default': '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 28c0-6 4-12 12-12s12 6 12 12v16c0 6-6 12-12 12s-12-6-12-12V28z" fill="#000000"/><path d="M24 20h4v8h-4zM36 20h4v8h-4zM28 36h8M32 40v6" stroke="#000000" stroke-width="2"/></svg>'
    };
    
    return icons[word] || icons['default'];
}

function playResultAudio(text) {
    speak(text);
}

/* ============================================
   FINGERSPELLING MODE FUNCTIONALITY
   ============================================ */

let fingerspellSessionId = null;
let fingerspellLanguage = null;
let fingerspellVideo = null;
let fingerspellCanvas = null;
let fingerspellCtx = null;
let fingerspellInterval = null;
let isFingerSpelling = false;
let detectedLetters = [];
let currentDetectedLetter = '';
let letterConfirmCount = 0;
// Require more stability to confirm a letter (reduces rapid flicker)
const LETTER_CONFIRM_THRESHOLD = 5; // Need 5 consecutive same letters to confirm
let fingerspellFinalizeTimer = null;
const FINGERSPELL_FINALIZE_MS = 2600; // inactivity -> finalize word
let predictionStartedByFingerspell = false;
// Debug helpers
let fingerspellDebugEl = null;
let fingerspellFramesSent = 0;
let fingerspellLastResponse = null;
let gestureLastResponse = null;

function updateGlobalDebug() {
    // Debug panel intentionally disabled for end-users.
    // (We keep internal state variables for development, but don't render them.)
    return;
}

// Initialize fingerspelling elements
function initFingerspellElements() {
    fingerspellVideo = document.getElementById('fingerspell-video');
    fingerspellCanvas = document.getElementById('fingerspell-canvas');
    if (fingerspellCanvas) {
        fingerspellCtx = fingerspellCanvas.getContext('2d');
        fingerspellCanvas.width = 640;
        fingerspellCanvas.height = 480;
    }

    // Debug panel intentionally disabled for end-users.
}

// Select fingerspelling language
function selectFingerspellLanguage(lang) {
    fingerspellLanguage = lang;
    
    // Update UI
    document.querySelectorAll('.fingerspell-lang-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === lang);
    });
    
    // Show the language badge
    const badge = document.getElementById('fingerspell-lang-badge');
    if (badge) {
        badge.textContent = lang;
        badge.style.display = 'block';
    }
    
    // Enable start button
    const startBtn = document.getElementById('start-fingerspell-camera');
    if (startBtn) {
        startBtn.disabled = false;
    }
    
    console.log('Fingerspell language selected:', lang);
}

// Start fingerspelling camera
async function startFingerspellCamera() {
    if (!fingerspellLanguage) {
        alert('Please select a sign language first');
        return;
    }
    
    initFingerspellElements();
    
    try {
        // Start a new session
        const sessionRes = await Promise.resolve({ok:true, json:()=>Promise.resolve({success:false})});
        const sessionData = await sessionRes.json();
        
        if (sessionData.success) {
            fingerspellSessionId = sessionData.session_id;
            console.log('Fingerspell session started:', fingerspellSessionId);
        }
        
        // Start camera (shared stream)
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 }
        });

        // Attach stream to fingerspell video and also to global video used by gesture prediction
        fingerspellVideo.srcObject = stream;
        await fingerspellVideo.play();

        // Make the gesture system use the same video/canvas so both detectors run on the same frames
        video = fingerspellVideo;
        canvas = fingerspellCanvas;
        ctx = fingerspellCtx;
        // Resize global canvas to match fingerspell canvas
        if (canvas) {
            canvas.width = fingerspellCanvas.width;
            canvas.height = fingerspellCanvas.height;
        }
        
        // Update UI
        document.getElementById('fingerspell-camera-placeholder').classList.add('hidden');
        document.getElementById('start-fingerspell-camera').classList.add('hidden');
        document.getElementById('stop-fingerspell-camera').classList.remove('hidden');
        document.getElementById('stop-fingerspell-camera').disabled = false;
        
        isFingerSpelling = true;
        detectedLetters = [];
        updateDetectedLettersUI();
        
        // Start detection loop for fingerspelling
    fingerspellInterval = setInterval(captureAndDetectLetter, FINGERSPELL_PREDICT_INTERVAL_MS);
        console.log('Fingerspell camera started');
    // Debug
    fingerspellFramesSent = 0;
    fingerspellLastResponse = null;
    updateFingerspellDebug();

        // Also ensure gesture prediction runs on the same frames. If not already running, start it.
        if (!predictionInterval) {
            predictionInterval = setInterval(captureAndPredict, GESTURE_PREDICT_INTERVAL_MS);
            predictionStartedByFingerspell = true;
            isRecording = true;
        }
        
    } catch (err) {
        console.error('Fingerspell camera error:', err);
        alert('Could not access camera');
    }
}

// Stop fingerspelling camera
function stopFingerspellCamera() {
    isFingerSpelling = false;
    
    if (fingerspellVideo && fingerspellVideo.srcObject) {
        fingerspellVideo.srcObject.getTracks().forEach(t => t.stop());
        fingerspellVideo.srcObject = null;
    }
    
    if (fingerspellInterval) {
        clearInterval(fingerspellInterval);
        fingerspellInterval = null;
    }

    // If this session started the gesture prediction, stop that too
    if (predictionStartedByFingerspell && predictionInterval) {
        clearInterval(predictionInterval);
        predictionInterval = null;
        predictionStartedByFingerspell = false;
        isRecording = false;
    }
    
    // Update UI
    document.getElementById('fingerspell-camera-placeholder').classList.remove('hidden');
    document.getElementById('start-fingerspell-camera').classList.remove('hidden');
    document.getElementById('stop-fingerspell-camera').classList.add('hidden');
    
    // Clear current letter display
    const currentLetterEl = document.getElementById('current-detected-letter');
    if (currentLetterEl) {
        currentLetterEl.textContent = '-';
    }
    
    console.log('Fingerspell camera stopped');
}

// Capture frame and detect letter
async function captureAndDetectLetter() {
    if (!isFingerSpelling || !fingerspellVideo || !fingerspellCtx) return;
    
    try {
        // Draw frame to canvas
        fingerspellCtx.drawImage(fingerspellVideo, 0, 0, fingerspellCanvas.width, fingerspellCanvas.height);
        
        // Get image data
        const imageData = fingerspellCanvas.toDataURL('image/jpeg', 0.8);
        
        // Send to detection API (backend expects /frame)
        const res = await Promise.resolve({ok:true, json:()=>Promise.resolve({success:false})});
        
        // Count frames sent for debug
        fingerspellFramesSent++;

        let result = null;
        try {
            result = await res.json();
        } catch (je) {
            // non-json response
            result = { error: 'Invalid JSON response', status: res.status };
        }
        fingerspellLastResponse = result;
        updateFingerspellDebug();

        // Backend returns current_letter/current_confidence and detected_letters
        // Update UI from backend detector state if available
        const letter = result.current_letter || null;
        const confidence = (result.current_confidence || result.current_confidence === 0) ? result.current_confidence : null;

        // Use backend's detected_letters to keep frontend in sync
        if (result.detected_letters && result.detected_letters.length > 0) {
            detectedLetters = result.detected_letters.slice();
            updateDetectedLettersUI();
            updateDetectedNameUI();
        } else if (result.partial_name && result.partial_name.length > 0) {
            // Backend provides a partial_name (string) — show it as intermediate feedback
            detectedLetters = result.partial_name.split('');
            updateDetectedLettersUI();
            updateDetectedNameUI();
        }

        // Show transient current_letter if present
        if (letter) {
            updateCurrentLetter(letter, confidence);
        }

        // If backend reports completion (final_result), use it to finalize the word immediately
        if (result.is_complete && result.final_result && result.final_result.detected_name) {
            const finalName = result.final_result.detected_name;
            if (finalName && finalName.length > 0) {
                // Add finalized name to sentence and clear local buffer
                addToSentence(finalName);
                detectedLetters = [];
                updateDetectedLettersUI();
                updateDetectedNameUI();
                if (fingerspellFinalizeTimer) { clearTimeout(fingerspellFinalizeTimer); fingerspellFinalizeTimer = null; }
            }
        }
        
    } catch (err) {
        console.error('Letter detection error:', err);
        if (fingerspellDebugEl) fingerspellDebugEl.innerText = 'Letter detection error: ' + err;
    }
}

function updateFingerspellDebug() {
    // Debug panel intentionally disabled for end-users.
    return;
}

// Update current detected letter with confirmation logic
function updateCurrentLetter(letter, confidence) {
    const currentLetterEl = document.getElementById('current-detected-letter');
    const microCurrentEl = document.getElementById('micro-current-detected-letter');

    // Priority logic: if a full WORD is currently being detected, do not process alphabets.
    if (recognitionMode === MODE_WORD) {
        if (currentLetterEl) { currentLetterEl.textContent = '-'; currentLetterEl.style.opacity = 0.6; }
        if (microCurrentEl) { microCurrentEl.textContent = '-'; microCurrentEl.style.opacity = 0.6; }
        // Also reset smoothing so we don't "catch up" with stale letters when switching modes.
        currentDetectedLetter = '';
        letterConfirmCount = 0;
        return;
    }

    // Ignore BLANK noise unless very confident
    try {
        if (typeof letter === 'string' && letter.trim().toUpperCase() === 'BLANK') {
            if (typeof confidence !== 'number' || confidence < BLANK_CONFIDENCE_THRESHOLD) {
                return;
            }
        }
    } catch (e) {}

    // Ignore low-confidence letters (prevents random letters when hand isn't stable)
    if (typeof confidence === 'number' && confidence < LETTER_CONFIDENCE_THRESHOLD) {
        return;
    }

    // Temporal smoothing for letters (separate from backend): only commit after N stable frames
    const letterStr = (typeof letter === 'string') ? letter.trim() : '';
    if (!letterStr) return;
    if (stableLetterLabel === letterStr) {
        stableLetterCount++;
    } else {
        stableLetterLabel = letterStr;
        stableLetterCount = 1;
    }

    if (stableLetterCount < STABLE_LETTER_FRAMES_REQUIRED) {
        // Show transient letter but don't commit yet
        if (currentLetterEl) {
            currentLetterEl.textContent = letterStr;
            currentLetterEl.style.opacity = 0.4 + (stableLetterCount * 0.15);
        }
        if (microCurrentEl) {
            microCurrentEl.textContent = letterStr;
            microCurrentEl.style.opacity = 0.4 + (stableLetterCount * 0.15);
        }
        return;
    }

    // Commit (with cooldown to avoid duplicates)
    const now = Date.now();
    const canCommit = (lastCommittedLetter !== letterStr) || (now - lastCommittedLetterAt > LETTER_COMMIT_COOLDOWN_MS);
    if (canCommit) {
        addDetectedLetter(letterStr);
        lastCommittedLetter = letterStr;
        lastCommittedLetterAt = now;
    }

    // Reset classic confirm logic to avoid double-adding
    currentDetectedLetter = '';
    letterConfirmCount = 0;

    if (currentLetterEl) {
        currentLetterEl.textContent = '✓';
        setTimeout(() => { currentLetterEl.textContent = '-'; }, 300);
    }
    if (microCurrentEl) {
        microCurrentEl.textContent = '✓';
        setTimeout(() => { microCurrentEl.textContent = '-'; }, 300);
    }
    return;
    
    if (letter === currentDetectedLetter) {
        letterConfirmCount++;
        
        if (letterConfirmCount >= LETTER_CONFIRM_THRESHOLD) {
            // Confirmed letter - add to detected letters
            addDetectedLetter(letter);
            currentDetectedLetter = '';
            letterConfirmCount = 0;
            
            if (currentLetterEl) {
                currentLetterEl.textContent = '✓';
                setTimeout(() => { currentLetterEl.textContent = '-'; }, 300);
            }
            if (microCurrentEl) {
                microCurrentEl.textContent = '✓';
                setTimeout(() => { microCurrentEl.textContent = '-'; }, 300);
            }
        } else {
            // Building confirmation
            if (currentLetterEl) {
                currentLetterEl.textContent = letter;
                currentLetterEl.style.opacity = 0.5 + (letterConfirmCount * 0.25);
            }
            if (microCurrentEl) {
                microCurrentEl.textContent = letter;
                microCurrentEl.style.opacity = 0.5 + (letterConfirmCount * 0.25);
            }
        }
    } else {
        // New letter detected
        currentDetectedLetter = letter;
        letterConfirmCount = 1;
        
        if (currentLetterEl) {
            currentLetterEl.textContent = letter;
            currentLetterEl.style.opacity = 0.5;
        }
        if (microCurrentEl) {
            microCurrentEl.textContent = letter;
            microCurrentEl.style.opacity = 0.5;
        }
    }
}

// Add confirmed letter to the list
function addDetectedLetter(letter) {
    // Prevent duplicate letters being added repeatedly
    try {
        const last = detectedLetters && detectedLetters.length > 0 ? detectedLetters[detectedLetters.length - 1] : null;
        if (last && String(last).toUpperCase() === String(letter).toUpperCase()) {
            return;
        }
    } catch (e) {}
    detectedLetters.push(letter);
    updateDetectedLettersUI();
    updateDetectedNameUI();
    
    // Visual feedback
    showFeedback('+ ' + letter);
    // Reset finalize timer (auto-convert letters -> word after short pause)
    if (fingerspellFinalizeTimer) clearTimeout(fingerspellFinalizeTimer);
    fingerspellFinalizeTimer = setTimeout(finalizeFingerspellWord, FINGERSPELL_FINALIZE_MS);
}

// Finalize current detected letters into a word and add to sentence
function finalizeFingerspellWord() {
    if (!detectedLetters || detectedLetters.length === 0) return;
    // Join letters into a word, collapse spaces
    const joined = detectedLetters.join('').trim();
    if (joined.length === 0) {
        clearFingerspellLetters();
        return;
    }

    // Add as a word to the sentence
    addToSentence(joined);
    // Clear letters after adding
    detectedLetters = [];
    currentDetectedLetter = '';
    letterConfirmCount = 0;
    updateDetectedLettersUI();
    updateDetectedNameUI();
    if (fingerspellFinalizeTimer) { clearTimeout(fingerspellFinalizeTimer); fingerspellFinalizeTimer = null; }
}

// Update detected letters display
function updateDetectedLettersUI() {
    // Update main grid if exists
    const container = document.getElementById('detected-letters-grid');
    const micro = document.getElementById('micro-detected-letters-grid');
    const htmlFor = (letters) => {
        if (!letters || letters.length === 0) return '<span style="color:#666;font-style:italic">Letters will appear here...</span>';
        let html = '';
        letters.forEach((letter, index) => {
            html += `<div class="letter" style="animation-delay: ${index * 0.05}s">${letter}</div>`;
        });
        return html;
    };
    if (container) container.innerHTML = htmlFor(detectedLetters);
    if (micro) micro.innerHTML = htmlFor(detectedLetters);
}

// Update detected name display
function updateDetectedNameUI() {
    const nameEl = document.getElementById('detected-name-text');
    const micro = document.getElementById('micro-detected-name-text');
    const text = (detectedLetters.length === 0) ? 'Your name will appear here' : detectedLetters.join('');
    if (nameEl) {
        nameEl.textContent = text;
        nameEl.classList.toggle('empty', detectedLetters.length === 0);
    }
    if (micro) {
        micro.textContent = text;
        micro.classList.toggle('empty', detectedLetters.length === 0);
    }
}

// Clear all detected letters
function clearFingerspellLetters() {
    detectedLetters = [];
    currentDetectedLetter = '';
    letterConfirmCount = 0;
    updateDetectedLettersUI();
    updateDetectedNameUI();
    
    const currentLetterEl = document.getElementById('current-detected-letter');
    if (currentLetterEl) {
        currentLetterEl.textContent = '-';
        currentLetterEl.style.opacity = 1;
    }
    if (fingerspellFinalizeTimer) { clearTimeout(fingerspellFinalizeTimer); fingerspellFinalizeTimer = null; }
}

// Add space between words
function addFingerspellSpace() {
    // Finalize current buffered letters into a word (user indicated a space)
    finalizeFingerspellWord();
    showFeedback('⎵ word boundary');
}

// Backspace - remove last letter
function fingerspellBackspace() {
    if (detectedLetters.length > 0) {
        detectedLetters.pop();
        updateDetectedLettersUI();
        updateDetectedNameUI();
        return;
    }
    // If no buffered letters, remove the last word from the sentence
    if (sentenceWords.length > 0) {
        sentenceWords.pop();
        renderSentence();
        showFeedback('removed last word');
    }
}

// Speak the detected name
function speakFingerspellName() {
    if (detectedLetters.length === 0) {
        alert('No name detected yet');
        return;
    }
    
    const name = detectedLetters.join('');
    speak(name);
    
    // Also show visual feedback
    showFeedback('🔊 ' + name);
}

// Go back from fingerspelling mode
function goBackFromFingerspell() {
    stopFingerspellCamera();
    clearFingerspellLetters();
    fingerspellLanguage = null;
    fingerspellSessionId = null;
    
    // Reset language selection
    document.querySelectorAll('.fingerspell-lang-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const badge = document.getElementById('fingerspell-lang-badge');
    if (badge) {
        badge.style.display = 'none';
    }
    
    // Hide fingerspell section, show mode selection
    document.getElementById('fingerspell-section').classList.add('hidden');
    document.getElementById('mode-selection').classList.remove('hidden');
    document.getElementById('language-section').classList.remove('hidden');
    
    currentMode = null;
}

// Update selectMode to handle fingerspell mode
const originalSelectMode = selectMode;
selectMode = function(mode) {
    if (mode === 'fingerspell') {
        currentMode = 'fingerspell';
        document.getElementById('mode-selection').classList.add('hidden');
        document.getElementById('language-section').classList.add('hidden');
        document.getElementById('fingerspell-section').classList.remove('hidden');
        
        // Disable start camera until language is selected
        const startBtn = document.getElementById('start-fingerspell-camera');
        if (startBtn) {
            // Auto-select a sensible default (ASL) so the user can start the camera immediately
            selectFingerspellLanguage('ASL');
            startBtn.disabled = false;
        }
        
        return;
    }
    
    originalSelectMode(mode);
};

var style = document.createElement('style');
// Black and white theme styling
style.textContent = '\n  @keyframes popFade{0%{opacity:1;transform:translate(-50%,-50%) scale(1)}100%{opacity:0;transform:translate(-50%,-80%) scale(0.9)}}\n  .word-chip{display:inline-flex;align-items:center;gap:8px;background:#000000;color:white;padding:8px 14px;border-radius:18px;margin:5px;font-weight:600} \n  .chip-remove{background:rgba(255,255,255,0.2);border:none;color:white;width:22px;height:22px;border-radius:50%;cursor:pointer;font-size:14px} \n  .chip-remove:hover{background:rgba(255,255,255,0.3)}\n  .gesture-sequence{display:flex;flex-wrap:wrap;gap:16px;justify-content:center;margin-top:12px} \n  .gesture-card{background:#ffffff;border:2px solid #e0e0e0;border-radius:12px;padding:20px;min-width:160px;max-width:200px;text-align:center;position:relative;color:#000000;box-shadow:0 2px 8px rgba(0,0,0,0.06);transition:all 0.2s ease} \n  .gesture-card:hover{border-color:#000000} \n  .gesture-card.fingerspell{background:#f5f5f5;border-color:#000000} \n  .gesture-card.active{border-color:#000000;border-width:2px} \n  .card-num{position:absolute;top:-12px;left:50%;transform:translateX(-50%);background:#000000;color:white;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem} \n  .card-word{font-size:1rem;font-weight:700;color:#000000;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.6px} \n  .card-icon{width:64px;height:64px;margin:8px auto;display:flex;align-items:center;justify-content:center} \n  .card-icon svg{width:100%;height:100%} \n  .card-desc{color:#666666;font-size:0.85rem;margin-top:8px;min-height:36px;line-height:1.4} \n  .card-hint{color:#000000;font-size:0.8rem;margin-top:6px;font-weight:600} \n  .fingerspell-letters{display:flex;gap:6px;justify-content:center;flex-wrap:wrap;margin:10px 0} \n  .letter-badge{background:#000000;color:white;width:36px;height:36px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:1rem} \n';
document.head.appendChild(style);

setInterval(checkBackendHealth, 30000);
console.log('GestureX app loaded');
