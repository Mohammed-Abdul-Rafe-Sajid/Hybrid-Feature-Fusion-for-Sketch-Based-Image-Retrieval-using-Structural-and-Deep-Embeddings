import os
import shutil
from torchvision.datasets import STL10
import numpy as np
import cv2

# Load dataset
dataset = STL10(root="data/raw", split="train", download=False)

# Create folder
os.makedirs("data/sample/raw", exist_ok=True)

# Number of samples
N = 500

for i in range(N):
    img, label = dataset[i]

    # Convert to numpy
    img = np.array(img)

    # Save image
    cv2.imwrite(f"data/sample/raw/img_{i}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print("Sample dataset created!")