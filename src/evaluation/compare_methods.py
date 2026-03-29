import numpy as np
import cv2
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# ---------------- MODEL ----------------
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- LOAD DATA ----------------
raw_emb = np.load("features/raw_embeddings.npy")
edge_emb = np.load("features/edge_embeddings.npy")

fused_03 = np.load("features/fused_0.3.npy")
fused_05 = np.load("features/fused_0.5.npy")
fused_07 = np.load("features/fused_0.7.npy")
fused_09 = np.load("features/fused_0.9.npy")

filenames = np.load("features/filenames.npy", allow_pickle=True)

# dataset for labels
from torchvision.datasets import STL10
dataset = STL10(root="data/raw", split="train", download=False)

# ---------------- FUNCTIONS ----------------

def extract_query_embedding(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        emb = model(img)

    return emb.view(-1).numpy()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def precision_at_k(embeddings, query_idx, k=5):
    query_file = filenames[query_idx]
    query_path = f"data/sample/raw/{query_file}"

    query_emb = extract_query_embedding(query_path)

    similarities = []

    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_emb, emb)
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    _, true_label = dataset[query_idx]

    correct = 0

    for i in range(k):
        idx = similarities[i][0]
        _, label = dataset[idx]

        if label == true_label:
            correct += 1

    return correct / k


def evaluate(embeddings, name, k_values=[3, 5, 10], num_samples=200):
    print(f"\n=== Evaluating {name} ===")

    results = {}

    for k in k_values:
        scores = []

        for i in range(num_samples):
            score = precision_at_k(embeddings, i, k=k)
            scores.append(score)

        mean = np.mean(scores)
        std = np.std(scores)

        results[k] = (mean, std)

        print(f"P@{k}: {mean:.3f} ± {std:.3f}")

    return results


# ---------------- RUN ----------------
all_results = {}

all_results["Raw"] = evaluate(raw_emb, "Raw")
all_results["Edge"] = evaluate(edge_emb, "Edge")
all_results["Fusion 0.3"] = evaluate(fused_03, "Fusion 0.3")
all_results["Fusion 0.5"] = evaluate(fused_05, "Fusion 0.5")
all_results["Fusion 0.7"] = evaluate(fused_07, "Fusion 0.7")
all_results["Fusion 0.9"] = evaluate(fused_09, "Fusion 0.9")

# ---------------- SAVE RESULTS ----------------
os.makedirs("reports/tables", exist_ok=True)

with open("reports/tables/comparison.txt", "w") as f:
    for method, k_results in all_results.items():
        f.write(f"\n{method}\n")
        for k, (mean, std) in k_results.items():
            f.write(f"P@{k}: {mean:.3f} ± {std:.3f}\n")

print("\nResults saved!")