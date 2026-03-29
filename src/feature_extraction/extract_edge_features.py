import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from model import get_model

# Paths
input_dir = "data/sample/edges"

# Load model
model = get_model()

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

features = []

files = os.listdir(input_dir)

for file in files:
    img_path = os.path.join(input_dir, file)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Convert to 3-channel
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        embedding = model(img)

    embedding = embedding.view(-1).numpy()

    features.append(embedding)

# Save
np.save("features/edge_embeddings.npy", np.array(features))

print("Edge embeddings extracted!")