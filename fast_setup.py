import os
import shutil
import numpy as np
from PIL import Image
import subprocess
import sys

def create_dummy_data():
    print("Creating dummy dataset directories...")
    base_dir = "data/raw/CASIA2"
    au_dir = os.path.join(base_dir, "Au")
    tp_dir = os.path.join(base_dir, "Tp")
    
    os.makedirs(au_dir, exist_ok=True)
    os.makedirs(tp_dir, exist_ok=True)
    
    print("Generating dummy images...")
    # Create a simple dummy image
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save Authentic images
    for i in range(2):
        img.save(os.path.join(au_dir, f"dummy_au_{i}.jpg"))
        
    # Save Tampered images (must contain 'Sp' or 'CM' in filename)
    for i in range(2):
        img.save(os.path.join(tp_dir, f"dummy_Sp_{i}.jpg"))
        
    print("Dummy data created successfully.")

def run_training():
    print("Running training script...")
    # Assuming we are in a virtual environment, but fallback to python
    python_cmd = sys.executable
    try:
        subprocess.run([python_cmd, "src/train.py", "--epochs", "1", "--batch-size", "2"], check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")

if __name__ == "__main__":
    create_dummy_data()
    run_training()
