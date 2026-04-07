import numpy as np
import pandas as pd
import os

def generate_asl_word_data(output_path: str, samples_per_class: int = 200):
    words = ["HELLO", "YES", "NO", "THANK YOU"]
    data = []

    for word in words:
        for _ in range(samples_per_class):
            landmarks = np.zeros((21, 3))
            landmarks[0] = [0, 0, 0] # Wrist
            
            # Simple logic for common ASL words:
            # HELLO: flat palm up
            # YES: fist nodding (just fist in one position here)
            # NO: index/middle/thumb touch
            # THANK YOU: flat palm near chin (landmark relative to wrist)

            if word == "HELLO":
                for finger in range(1, 6): # All fingers extended
                    landmarks[1 + (finger-1)*4 + 3] = [finger * 0.1, 0.8, 0]
            elif word == "YES":
                for finger in range(1, 6): # Closed fist
                    landmarks[1 + (finger-1)*4 + 3] = [finger * 0.05, 0.1, 0]
            elif word == "NO":
                # Thumb, Index, Middle tips together
                landmarks[4] = [0.1, 0.4, 0] # Thumb
                landmarks[8] = [0.1, 0.4, 0] # Index
                landmarks[12] = [0.1, 0.4, 0] # Middle
            elif word == "THANK YOU":
                # Flat palm leaning 
                for finger in range(1, 6):
                    landmarks[1 + (finger-1)*4 + 3] = [0, 0.8, -0.2]

            # Add jitter
            landmarks += np.random.normal(0, 0.03, landmarks.shape)
            
            row = landmarks.flatten().tolist() + [word]
            data.append(row)

    df = pd.DataFrame(data)
    df.columns = [f"f{i}" for i in range(63)] + ["label"]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} samples for ASL Words -> {output_path}")

if __name__ == "__main__":
    generate_asl_word_data("/Users/chandrurajinikanth/Desktop/gesture project/GestureX/data/landmarks/ASL/word_data.csv")
