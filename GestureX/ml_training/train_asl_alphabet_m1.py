"""
train_asl_alphabet_m1.py
========================
Trains an ASL Alphabet classifier (29 classes: A-Z + Space, Delete, Nothing)
on Apple M1/M2/M3 using PyTorch Metal Performance Shaders (MPS) acceleration.

Dataset  : ASL Alphabet (Kaggle — 87,000 images)
           kaggle.com/datasets/grassknoted/asl-alphabet
Backbone : MobileNetV3-Large (pretrained on ImageNet)
Output   : asl_alphabet_m1.pth   — full model weights
           asl_alphabet_m1.onnx  — ONNX export for browser/backend use

Usage:
  python train_asl_alphabet_m1.py
  python train_asl_alphabet_m1.py --epochs 20 --batch_size 64 --lr 0.0005

Requirements (install once):
  pip install torch torchvision kagglehub Pillow tqdm
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # macOS cert fix

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ─── CLI ARGS ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train ASL Alphabet classifier on M1")
parser.add_argument("--dataset_path", type=str, default="",
    help="Path to ASL alphabet training folder (auto-downloaded if empty)")
parser.add_argument("--epochs",     type=int,   default=15)
parser.add_argument("--batch_size", type=int,   default=32)
parser.add_argument("--lr",         type=float, default=0.001)
parser.add_argument("--val_split",  type=float, default=0.10,
    help="Fraction of data reserved for validation (default 10%%)")
parser.add_argument("--workers",    type=int,   default=0)
parser.add_argument("--output_dir", type=str,   default="../models/ASL",
    help="Directory to save weights and metadata")
parser.add_argument("--no_onnx",    action="store_true",
    help="Skip ONNX export after training")
args = parser.parse_args()

# ─── 1. DEVICE SETUP (M1 MPS > CUDA > CPU) ──────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU (MPS) — M1/M2/M3 acceleration active")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU — MPS not available. Training will be slow.")

# ─── 2. DATASET PATH ─────────────────────────────────────────
dataset_path = args.dataset_path

if not dataset_path:
    print("\n📦 No dataset path provided — attempting auto-download via kagglehub...")
    try:
        import kagglehub
        # Download the ASL Alphabet dataset (grassknoted/asl-alphabet)
        download_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
        print(f"   Downloaded to: {download_path}")

        # The dataset usually has a nested 'asl_alphabet_train/asl_alphabet_train' structure
        candidates = [
            Path(download_path) / "asl_alphabet_train" / "asl_alphabet_train",
            Path(download_path) / "asl_alphabet_train",
            Path(download_path),
        ]
        for candidate in candidates:
            if candidate.exists() and any(candidate.iterdir()):
                dataset_path = str(candidate)
                break

        if not dataset_path:
            sys.exit("❌ Could not locate image folder inside the downloaded dataset.")

        print(f"✅ Using dataset at: {dataset_path}")

    except ImportError:
        sys.exit(
            "❌ kagglehub not installed. Run:\n"
            "   pip install kagglehub\n"
            "Or pass --dataset_path /path/to/asl_alphabet_train"
        )
    except Exception as e:
        sys.exit(
            f"❌ Auto-download failed: {e}\n"
            "Manual fix: Download from kaggle.com/datasets/grassknoted/asl-alphabet\n"
            "then run:  python train_asl_alphabet_m1.py --dataset_path /path/to/asl_alphabet_train"
        )

if not os.path.isdir(dataset_path):
    sys.exit(f"❌ Dataset path does not exist: {dataset_path}")

# ─── 3. TRANSFORMS ───────────────────────────────────────────
# Training: aggressive augmentation for webcam robustness
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),      # webcams are often mirrored
    transforms.RandomRotation(degrees=15),         # different signing angles
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Validation: clean, no augmentation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ─── 4. DATA LOADING ─────────────────────────────────────────
print(f"\n📂 Loading dataset from: {dataset_path}")
full_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transforms)
num_classes  = len(full_dataset.classes)
class_names  = full_dataset.classes

print(f"   Classes ({num_classes}): {class_names}")
print(f"   Total images: {len(full_dataset):,}")

# Train / validation split
val_size   = int(len(full_dataset) * args.val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Override val transforms (random_split shares the same dataset object)
val_dataset.dataset = datasets.ImageFolder(root=dataset_path, transform=val_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(args.workers > 0),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(args.workers > 0),
)

print(f"   Train: {train_size:,} images  |  Val: {val_size:,} images")
print(f"   Batch size: {args.batch_size}  |  Batches per epoch: {len(train_loader)}")

# ─── 5. MODEL ARCHITECTURE ───────────────────────────────────
print("\n🧠 Building MobileNetV3-Large (pretrained ImageNet)...")

model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# Replace the classifier head to match num_classes
# MobileNetV3-Large classifier: Sequential(Linear, HardSwish, Dropout, Linear)
in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.Hardswish(),
    nn.Dropout(p=0.3),
    nn.Linear(512, num_classes),
)

# Freeze all backbone layers for first 5 epochs (feature extraction phase)
# Then unfreeze for fine-tuning
for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)

total_params   = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total params:     {total_params:,}")
print(f"   Trainable params: {trainable_params:,} (backbone frozen for warm-up)")

# ─── 6. LOSS + OPTIMISER + SCHEDULER ─────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# Only optimise the new head during warm-up
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=1e-4,
)

scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

# ─── 7. OUTPUT DIRECTORY ─────────────────────────────────────
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
weights_path = output_dir / "asl_alphabet_m1.pth"
best_path    = output_dir / "asl_alphabet_m1_best.pth"
meta_path    = output_dir / "asl_alphabet_m1_meta.json"

# ─── 8. TRAINING LOOP ────────────────────────────────────────
print(f"\n🚀 Starting training for {args.epochs} epochs on {device}")
print("="*65)

UNFREEZE_EPOCH = 5   # Unfreeze backbone after this many epochs
best_val_acc   = 0.0
history        = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, args.epochs + 1):
    epoch_start = time.time()

    # Unfreeze backbone for fine-tuning after warm-up
    if epoch == UNFREEZE_EPOCH + 1:
        print(f"\n🔓 Epoch {epoch}: Unfreezing backbone for full fine-tuning...")
        for param in model.features.parameters():
            param.requires_grad = True
        # Replace optimizer to include all params at a lower LR
        optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch, eta_min=1e-6)

    # ── Train ────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total   = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{args.epochs} [TRAIN]",
                leave=False, ncols=90)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * images.size(0)
        preds          = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            running_acc  = 100.0 * train_correct / train_total
            running_loss = train_loss / train_total
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.1f}%")

    epoch_train_loss = train_loss / train_total
    epoch_train_acc  = 100.0 * train_correct / train_total

    # ── Validate ─────────────────────────────────────────────
    model.eval()
    val_loss    = 0.0
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch:2d}/{args.epochs} [VAL]  ",
                                   leave=False, ncols=90):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            val_loss    += loss.item() * images.size(0)
            preds        = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    epoch_val_loss = val_loss / val_total
    epoch_val_acc  = 100.0 * val_correct / val_total

    scheduler.step()
    elapsed = time.time() - epoch_start

    # ── Log ──────────────────────────────────────────────────
    history["train_loss"].append(epoch_train_loss)
    history["train_acc"].append(epoch_train_acc)
    history["val_loss"].append(epoch_val_loss)
    history["val_acc"].append(epoch_val_acc)

    print(
        f"Epoch {epoch:2d}/{args.epochs} | "
        f"Train Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.1f}% | "
        f"Val Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.1f}% | "
        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
        f"Time: {elapsed:.0f}s"
    )

    # ── Save best checkpoint ─────────────────────────────────
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "val_acc":     best_val_acc,
            "classes":     class_names,
        }, best_path)
        print(f"   ⭐ New best saved → {best_path.name}  (val_acc={best_val_acc:.1f}%)")

# ─── 9. SAVE FINAL WEIGHTS + METADATA ────────────────────────
print("\n💾 Saving final model weights...")
torch.save(model.state_dict(), weights_path)
print(f"   Final weights → {weights_path}")

meta = {
    "architecture":  "MobileNetV3-Large",
    "num_classes":   num_classes,
    "classes":       class_names,
    "input_size":    [224, 224],
    "normalize_mean":[0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "best_val_acc":  round(best_val_acc, 2),
    "epochs_trained":args.epochs,
    "dataset":       dataset_path,
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"   Metadata      → {meta_path}")
print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.1f}%")

# ─── 10. ONNX EXPORT ─────────────────────────────────────────
if not args.no_onnx:
    print("\n📤 Exporting to ONNX (for browser / FastAPI inference)...")
    try:
        onnx_path = output_dir / "asl_alphabet_m1.onnx"
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224, device=device)

        # ONNX export requires CPU for some ops on MPS
        if device.type == "mps":
            model_cpu   = model.to("cpu")
            dummy_input = dummy_input.to("cpu")
        else:
            model_cpu   = model

        torch.onnx.export(
            model_cpu,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"   ONNX model    → {onnx_path}")

        # Verify with onnxruntime if available
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path))
            dummy_np = dummy_input.numpy()
            out = sess.run(None, {"image": dummy_np})
            print(f"   ✅ ONNX verified — output shape: {out[0].shape}")
        except ImportError:
            print("   ℹ️  Install onnxruntime to verify: pip install onnxruntime")

    except Exception as e:
        print(f"   ⚠️  ONNX export failed: {e}")
        print("      (This doesn't affect the .pth weights — they are saved.)")

# ─── 11. INFERENCE TEST ──────────────────────────────────────
print("\n🔬 Quick inference test on a random batch...")
model.eval()
test_images, test_labels = next(iter(val_loader))
test_images = test_images.to(device)
with torch.no_grad():
    logits = model(test_images)
    probs  = torch.softmax(logits, dim=1)
    preds  = probs.argmax(dim=1).cpu()
    confs  = probs.max(dim=1).values.cpu()

print(f"   Sample predictions (first 8):")
for i in range(min(8, len(preds))):
    true_lbl = class_names[test_labels[i]]
    pred_lbl = class_names[preds[i]]
    conf     = confs[i].item() * 100
    status   = "✅" if true_lbl == pred_lbl else "❌"
    print(f"   {status} True: {true_lbl:8s}  Pred: {pred_lbl:8s}  Conf: {conf:5.1f}%")

print("\n" + "="*65)
print("🎯 TRAINING SUMMARY")
print("="*65)
print(f"   Architecture   : MobileNetV3-Large (ImageNet pretrained)")
print(f"   Classes        : {num_classes} ({', '.join(class_names[:5])}...)")
print(f"   Best Val Acc   : {best_val_acc:.1f}%")
print(f"   Weights saved  : {weights_path}")
print(f"   Best model     : {best_path}")
print(f"   Metadata       : {meta_path}")
if not args.no_onnx:
    print(f"   ONNX export    : {output_dir / 'asl_alphabet_m1.onnx'}")
print("\n📌 Next steps:")
print("   1. Load asl_alphabet_m1.pth into your GestureX backend inference")
print("   2. Use asl_alphabet_m1.onnx for the ONNX inference pipeline")
print("   3. Repeat with ISL/BSL/JSL folders for multi-language support")
print("="*65)
