import cv2
import os
import numpy as np

# Input & output paths
input_dir = "data/sample/raw"
output_dir = "data/sample/edges"

os.makedirs(output_dir, exist_ok=True)

files = os.listdir(input_dir)

for i, file in enumerate(files):
    img_path = os.path.join(input_dir, file)

    # Read image
    img = cv2.imread(img_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
# cv2.Canny(gray, 50, 150)
# cv2.Canny(gray, 100, 200)
    # Save edge image
    cv2.imwrite(os.path.join(output_dir, file), edges)

print(f"Processed {len(files)} images into edges!")