#!/usr/bin/env python3
"""
cleanup_workspace.py
====================
Safe ML workspace cleanup script for GestureX project.

Actions:
  1. Audits all .pth / .pt / .h5 / .weights files in the project
  2. Deletes old model files — PRESERVES asl_alphabet_m1.pth + asl_alphabet_m1_best.pth
  3. Optionally clears the kagglehub dataset cache (~1 GB)
  4. Reports disk space freed

Usage:
  python3 cleanup_workspace.py           # dry run — shows what WOULD be deleted
  python3 cleanup_workspace.py --confirm # actually deletes
  python3 cleanup_workspace.py --confirm --clear-kaggle  # also wipes dataset cache
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent          # GestureX/
SEARCH_EXTS    = {".pth", ".pt", ".h5", ".weights"}

# Files to ALWAYS preserve (never delete)
PROTECTED_NAMES = {
    "asl_alphabet_m1.pth",
    "asl_alphabet_m1_best.pth",
}

# Directories to skip entirely (venv test fixtures, torch cache, etc.)
SKIP_DIRS = {
    ".venv", "venv", "env",
    "__pycache__",
    "site-packages",
    "node_modules",
}

KAGGLE_CACHE = Path.home() / ".cache" / "kagglehub"
TORCH_CACHE  = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--confirm",      action="store_true",
    help="Actually delete files (default is dry-run)")
parser.add_argument("--clear-kaggle", action="store_true",
    help="Also clear the kagglehub dataset cache (~1 GB)")
parser.add_argument("--clear-torch",  action="store_true",
    help="Also clear PyTorch pretrained weight cache")
args = parser.parse_args()

DRY_RUN = not args.confirm

if DRY_RUN:
    print("🔍 DRY RUN — no files will be deleted (pass --confirm to execute)")
else:
    print("⚡ LIVE RUN — files will be permanently deleted")
print()

# ── Helper ────────────────────────────────────────────────────
def hr(): print("─" * 65)

def human_size(path):
    try:
        size = os.path.getsize(path)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    except Exception:
        return "?"

def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return False

# ── Step 1: Scan project for model files ─────────────────────
hr()
print("📂 Scanning project for model files...")
hr()

to_delete = []
to_keep   = []

for ext in SEARCH_EXTS:
    for f in PROJECT_ROOT.rglob(f"*{ext}"):
        if should_skip(f):
            continue
        name = f.name
        if name in PROTECTED_NAMES:
            to_keep.append(f)
        else:
            to_delete.append(f)

# Report keeps
print(f"\n✅ Protected (will NOT delete):")
for f in to_keep:
    print(f"   KEEP  [{human_size(f):>8s}]  {f.relative_to(PROJECT_ROOT.parent)}")

# Report deletions
print(f"\n🗑️  To delete ({len(to_delete)} file{'s' if len(to_delete) != 1 else ''}):")
total_freed = 0
for f in to_delete:
    sz = os.path.getsize(f) if f.exists() else 0
    total_freed += sz
    print(f"   DEL   [{human_size(f):>8s}]  {f.relative_to(PROJECT_ROOT.parent)}")

if not to_delete:
    print("   (none — workspace is already clean)")

# ── Step 2: Delete ───────────────────────────────────────────
if to_delete:
    hr()
    freed = 0
    for f in to_delete:
        sz = os.path.getsize(f) if f.exists() else 0
        if DRY_RUN:
            print(f"   [DRY RUN] Would delete: {f.name}")
        else:
            try:
                f.unlink()
                freed += sz
                print(f"   ✅ Deleted: {f.name}  ({human_size(f) if False else ''})")
            except Exception as e:
                print(f"   ❌ Failed:  {f.name}  ({e})")

    if not DRY_RUN:
        print(f"\n   Freed from project: {freed / 1024**2:.1f} MB")

# ── Step 3: kagglehub cache ──────────────────────────────────
if args.clear_kaggle:
    hr()
    print("🧹 Clearing kagglehub dataset cache...")
    if KAGGLE_CACHE.exists():
        cache_size = sum(
            f.stat().st_size for f in KAGGLE_CACHE.rglob("*") if f.is_file()
        )
        print(f"   Cache size: {cache_size / 1024**3:.2f} GB  →  {KAGGLE_CACHE}")
        if DRY_RUN:
            print("   [DRY RUN] Would delete entire kagglehub cache")
        else:
            shutil.rmtree(KAGGLE_CACHE)
            print("   ✅ kagglehub cache cleared")
    else:
        print("   ℹ️  Cache folder not found — already clean")
else:
    print(f"\n💡 kagglehub cache not cleared.")
    if KAGGLE_CACHE.exists():
        try:
            cache_size = sum(
                f.stat().st_size for f in KAGGLE_CACHE.rglob("*") if f.is_file()
            )
            print(f"   Run with --clear-kaggle to free {cache_size / 1024**3:.2f} GB")
            print(f"   Location: {KAGGLE_CACHE}")
        except Exception:
            print(f"   Location: {KAGGLE_CACHE}")

# ── Step 4: PyTorch checkpoint cache ─────────────────────────
if args.clear_torch:
    hr()
    print("🧹 Clearing PyTorch pretrained weight cache...")
    if TORCH_CACHE.exists():
        cache_size = sum(
            f.stat().st_size for f in TORCH_CACHE.rglob("*") if f.is_file()
        )
        print(f"   Cache size: {cache_size / 1024**2:.1f} MB  →  {TORCH_CACHE}")
        if DRY_RUN:
            print("   [DRY RUN] Would clear torch hub checkpoints")
        else:
            shutil.rmtree(TORCH_CACHE)
            TORCH_CACHE.mkdir(parents=True)
            print("   ✅ PyTorch checkpoint cache cleared")
    else:
        print("   ℹ️  Torch cache not found")

# ── Summary ───────────────────────────────────────────────────
hr()
print("📊 DISK SUMMARY")
hr()
stat = shutil.disk_usage("/")
print(f"   Total:     {stat.total / 1024**3:.1f} GB")
print(f"   Used:      {stat.used  / 1024**3:.1f} GB")
print(f"   Free:      {stat.free  / 1024**3:.1f} GB")
if DRY_RUN and (to_delete or args.clear_kaggle):
    print(f"\n💡 Run with --confirm to execute the cleanup above.")
hr()
