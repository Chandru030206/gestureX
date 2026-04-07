import tensorflow as tf
import os
from pathlib import Path

def convert_to_tflite(model_path, output_path):
    print(f"Converting {model_path} to {output_path}...")
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print("Done.")

if __name__ == "__main__":
    ASL_DIR = "/Users/chandrurajinikanth/Desktop/gesture project/GestureX/models/ASL"
    alpha_path = os.path.join(ASL_DIR, "alphabet_model.keras")
    word_path = os.path.join(ASL_DIR, "word_model.keras")
    
    if os.path.exists(alpha_path):
        convert_to_tflite(alpha_path, os.path.join(ASL_DIR, "alphabet_model.tflite"))
    if os.path.exists(word_path):
        convert_to_tflite(word_path, os.path.join(ASL_DIR, "word_model.tflite"))
