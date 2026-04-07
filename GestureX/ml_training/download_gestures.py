import requests
import os
import time

# 50 most common sign language words
COMMON_WORDS = [
  "hello", "goodbye", "thank you", "please", "sorry",
  "yes", "no", "help", "stop", "wait",
  "good", "bad", "more", "less", "water",
  "food", "home", "work", "love", "hate",
  "happy", "sad", "angry", "tired", "sick",
  "name", "age", "where", "what", "how",
  "morning", "night", "today", "tomorrow", "yesterday",
  "mother", "father", "friend", "school", "hospital",
  "left", "right", "up", "down", "here",
  "come", "go", "eat", "drink", "sleep"
]

LANG_CODES = {
  'asl': 'en.us', 'bsl': 'en.gb', 'isl': 'hi.in',
  'jsl': 'ja.jp', 'auslan': 'en.au', 'lsf': 'fr.fr',
  'dgs': 'de.de', 'libras': 'pt.br', 'ksl': 'ko.kr',
  'csl': 'zh.cn'
}

def download_gesture(lang, word, save_dir):
  os.makedirs(save_dir, exist_ok=True)
  
  # Try SpreadTheSign API
  lang_code = LANG_CODES.get(lang)
  url = f"https://api.spreadthesign.com/v1/search/{lang_code}/{word}"
  
  try:
    res = requests.get(url, timeout=5)
    data = res.json()
    if data.get('results'):
      video_url = data['results'][0].get('video_url')
      if video_url:
        video_res = requests.get(video_url, timeout=10)
        filename = f"{word.lower().replace(' ', '_')}.mp4"
        with open(os.path.join(save_dir, filename), 'wb') as f:
          f.write(video_res.content)
        print(f"✓ {lang}/{word}")
        return True
  except Exception as e:
    print(f"✗ {lang}/{word}: {e}")
  return False

# Download all
if __name__ == "__main__":
    # Ensure we are in the GestureX project root when running this from shell or adjust path
    # For execution via run_command in root:
    base_path = "public/gestures"
    
    total = 0
    for lang in LANG_CODES:
      for word in COMMON_WORDS:
        save_dir = f"{base_path}/{lang}"
        success = download_gesture(lang, word, save_dir)
        if success:
          total += 1
        time.sleep(0.5)  # rate limit

    print(f"\nDownloaded {total} gesture videos")
