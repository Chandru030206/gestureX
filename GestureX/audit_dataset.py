import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURE THIS ---
DATA_PATH = "data/landmarks/ASL/word_data.csv"
SAVE_PATH = "data_audit_report.png"
# ----------------------

def run_audit():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Audit counts
    counts = df['label'].value_counts().sort_values()
    
    print("\n--- GESTURE DATA AUDIT ---")
    print(f"Total Samples: {len(df)}")
    print(f"Total Classes: {len(counts)}")
    print(f"Average Samples per Class: {len(df)/len(counts):.1f}")
    print("\nPer-Class Distribution:")
    
    colors = []
    for label, count in counts.items():
        status = ""
        if count < 200:
            status = " [CRITICAL — needs more data]"
            colors.append('red')
        else:
            colors.append('green')
        print(f"{label}: {count}{status}")

    # Plotting
    plt.figure(figsize=(12, 8))
    counts.plot(kind='bar', color=colors)
    plt.axhline(y=200, color='gray', linestyle='--', label='Critical Threshold (200)')
    plt.title("Class Distribution Audit")
    plt.xlabel("Gesture Label")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    
    print(f"\n✅ Audit complete. Bar chart saved to {SAVE_PATH}")
    
    critical_classes = [l for l, c in counts.items() if c < 200]
    if critical_classes:
        print(f"\n⚠️  Attention needed for: {', '.join(critical_classes)}")
    else:
        print("\n✅ All classes meet the 200-sample threshold.")

if __name__ == "__main__":
    run_audit()
