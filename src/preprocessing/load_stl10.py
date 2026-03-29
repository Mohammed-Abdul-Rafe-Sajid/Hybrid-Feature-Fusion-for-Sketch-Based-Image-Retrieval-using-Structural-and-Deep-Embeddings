import torchvision
import torchvision.transforms as transforms
import os

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/sample/raw", exist_ok=True)

# Transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load STL-10
dataset = torchvision.datasets.STL10(
    root="data/raw",
    split="train",
    download=True,
    transform=transform
)

print(f"Total images in dataset: {len(dataset)}")