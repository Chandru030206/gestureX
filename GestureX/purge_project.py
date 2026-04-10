import os
import shutil
import glob
from pathlib import Path

def purge_project():
    print("🚀 Starting Project Purge...")

    # 1. Data Cleanup
    # ~/.cache/kagglehub
    kaggle_cache = Path.home() / ".cache" / "kagglehub"
    if kaggle_cache.exists():
        print(f"🧹 Deleting Kaggle cache: {kaggle_cache}")
        shutil.rmtree(kaggle_cache)
    
    # Local /datasets/ folders
    # Not necessarily in the root, let's scan project directory for any folder named 'datasets'
    project_root = Path.cwd()
    for datasets_dir in project_root.rglob("datasets"):
        if datasets_dir.is_dir():
            print(f"🧹 Deleting datasets directory: {datasets_dir}")
            shutil.rmtree(datasets_dir)

    # 2. Model Cleanup
    # Scan for .pth, .h5, .pt files.
    # Keep only those ending in _m1.pth or similar final models as specified in requirements
    # Final models: e.g., german_m1.pth, korean_m1.pth
    model_extensions = ["*.pth", "*.h5", "*.pt"]
    for ext in model_extensions:
        for model_file in project_root.rglob(ext):
            # If the filename ending matches our current "final" naming convention (_m1.pth)
            if "_m1.pth" in model_file.name:
                print(f"✅ Keeping final model: {model_file.name}")
                continue
            
            # Additional logic to keep common final names if specified, 
            # otherwise delete as they are likely checkpoints/temp
            print(f"🗑️ Deleting checkpoint/temp model: {model_file}")
            model_file.unlink()

    # 3. UI Cleanup
    # Delete 'Old UI' folder if it exists
    for old_ui in project_root.rglob("Old UI"):
        if old_ui.is_dir():
            print(f"🧹 Deleting Old UI folder: {old_ui}")
            shutil.rmtree(old_ui)

    # 4. Environment Cleanup
    # Clear __pycache__ and .DS_Store
    for pycache in project_root.rglob("__pycache__"):
        if pycache.is_dir():
            print(f"🧹 Clearing pycache: {pycache}")
            shutil.rmtree(pycache)
            
    for ds_store in project_root.rglob(".DS_Store"):
        print(f"🧹 Deleting .DS_Store: {ds_store}")
        ds_store.unlink()

    print("\n✅ Project Purge Complete! Workspace is now clean and optimized.")

if __name__ == "__main__":
    purge_project()
