import numpy as np
import pandas as pd
import os
from pathlib import Path

def generate_asl_alphabet_data(output_path: str, samples_per_class: int = 200):
    """
    Generates synthetic landmark data for ASL Alphabet (A-Z).
    Each sample is 63 features (21 landmarks * 3 coordinates).
    Uses wrist-relative normalization.
    """
    classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    data = []

    # MediaPipe Landmark Indices:
    # 0: Wrist
    # 1-4: Thumb
    # 5-8: Index
    # 9-12: Middle
    # 13-16: Ring
    # 17-20: Pinky

    # Canonical handshapes (simplified x, y, z relative to wrist at 0,0,0)
    # y-axis is negative for "up" in standard screen coords, but here we treat +y as up for simplicity 
    # and then flip/jitter.
    
    canonical_shapes = {
        'A': [[0.2,0.1,0], [0.3,0.3,0], [0.35,0.4,0], [0.4,0.5,0], # Thumb
              [0.1,0.4,0], [0.1,0.5,0], [0.1,0.4,0], [0.1,0.3,0], # All other fingers folded
              [0,-0.1,0], [0,-0.15,0], [0,-0.1,0], [0,-0.05,0],
              [-0.1,-0.1,0], [-0.1,-0.15,0], [-0.1,-0.1,0], [-0.1,-0.05,0],
              [-0.2,-0.1,0], [-0.2,-0.15,0], [-0.2,-0.1,0], [-0.2,-0.05,0]],
        'B': [[-0.2,0.2,0], [-0.1,0.3,0], [0,0.35,0], [0.1,0.35,0], # Thumb across palm
              [0.2,0.6,0], [0.2,0.8,0], [0.2,0.9,0], [0.2,1.0,0], # All fingers extended up
              [0.05,0.6,0], [0.05,0.8,0], [0.05,0.9,0], [0.05,1.0,0],
              [-0.05,0.6,0], [-0.05,0.8,0], [-0.05,0.9,0], [-0.05,1.0,0],
              [-0.2,0.6,0], [-0.2,0.8,0], [-0.2,0.9,0], [-0.2,1.0,0]],
        # ... simplified: we'll use a loop to create variation
    }

    # Since definining all 26 manually is tedious, we'll use a programmatic approach
    # to simulate finger extensions.
    
    for char in classes:
        for _ in range(samples_per_class):
            landmarks = np.zeros((21, 3))
            landmarks[0] = [0, 0, 0] # Wrist
            
            # Simulate finger states (0: folded, 1: extended)
            # This is a very rough approximation to create a balanced dataset.
            states = {
                'A': [0, 0, 0, 0, 0], # All folded (Thumb out)
                'B': [0, 1, 1, 1, 1], # All up
                'C': [0.5, 0.5, 0.5, 0.5, 0.5], # Curved
                'D': [0, 1, 0, 0, 0], # Index up
                'E': [0, 0, 0, 0, 0], # Tight fist
                'F': [0, 0, 1, 1, 1], # Index/Thumb touch (Ok)
                'G': [1, 1, 0, 0, 0], # Thumb/Index point sideways
                'H': [0, 1, 1, 0, 0], # Index/Middle point sideways
                'I': [0, 0, 0, 0, 1], # Pinky up
                'K': [0.5, 1, 1, 0, 0], # Victory sign with thumb
                'L': [1, 1, 0, 0, 0], # L shape
                'M': [0, 0, 0, 0, 0], # Fingers over thumb
                'N': [0, 0, 0, 0, 0], 
                'O': [0.5, 0.5, 0.5, 0.5, 0.5], # O shape
                'P': [0.5, 1, 1, 0, 0], # K pointing down
                'Q': [1, 1, 0, 0, 0], 
                'R': [0, 1, 1, 0, 0], # Crossed fingers
                'S': [0, 0, 0, 0, 0], # Fist thumb center
                'T': [0, 0, 0, 0, 0],
                'U': [0, 1, 1, 0, 0], # Index/Middle up together
                'V': [0, 1, 1, 0, 0], # Victory
                'W': [0, 1, 1, 1, 0], # Three up
                'X': [0, 0.5, 0, 0, 0], # Hooked index
                'Y': [1, 0, 0, 0, 1], # Thumb/Pinky up
            }.get(char, [0, 1, 1, 1, 1])

            # Apply state to fingers
            for f_idx, state in enumerate(states):
                # 4 landmarks per finger
                base_idx = 1 + f_idx * 4
                for joint in range(4):
                    # Add some noise and variance
                    offset = state * 0.2 * (joint + 1)
                    noise = np.random.normal(0, 0.02, 3)
                    landmarks[base_idx + joint] = [f_idx * 0.1, offset, 0] + noise

            # Flatten and add label
            row = landmarks.flatten().tolist() + [char]
            data.append(row)

    df = pd.DataFrame(data)
    df.columns = [f"f{i}" for i in range(63)] + ["label"]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} samples for ASL Alphabet -> {output_path}")

if __name__ == "__main__":
    generate_asl_alphabet_data("data/landmarks/ASL/alphabet_data.csv")
