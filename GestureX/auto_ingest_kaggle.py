import opendatasets as od
import os
import pandas as pd
import glob
import json
from train_language_model import train_language

# --- FORCE AUTH ---
try:
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
        creds = json.load(f)
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        print(f"✅ Kaggle Auth Loaded for: {creds['username']}")
except Exception as e:
    print(f"⚠️ Warning: Could not auto-load kaggle.json: {e}")
# ------------------
DATASETS = {
    "BSL": "https://www.kaggle.com/datasets/muhammad70557/bsl-dataset",
    "ISL": "https://www.kaggle.com/datasets/gauravpathak/indian-sign-language-dataset",
    "ASL": "https://www.kaggle.com/datasets/ayuraj/asl-dataset" # High quality backup
}
# -----------------------

def download_and_train():
    if not os.path.exists("data"): os.makedirs("data")
    
    for lang, url in DATASETS.items():
        print(f"📦 Downloading {lang} from Kaggle...")
        try:
            od.download(url, data_dir="data")
            
            # Find the CSV in the downloaded folder
            folder_name = url.split('/')[-1]
            csv_files = glob.glob(f"data/{folder_name}/**/*.csv", recursive=True)
            
            if csv_files:
                # Rename the found CSV to our standard format
                target_path = f"data/{lang}_gestures.csv"
                print(f"✨ Found CSV: {csv_files[0]}. Syncing to {target_path}")
                os.rename(csv_files[0], target_path)
                
                # Trigger internal professional training
                train_language(lang)
            else:
                print(f"⚠️ No CSV found for {lang}. Ensure the dataset contains a landmark file.")
        except Exception as e:
            print(f"❌ Error with {lang}: {e}")

if __name__ == "__main__":
    download_and_train()
