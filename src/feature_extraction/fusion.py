import numpy as np
import os

# Load embeddings
raw = np.load("features/raw_embeddings.npy")
edge = np.load("features/edge_embeddings.npy")

# Check shapes
print("Raw shape:", raw.shape)
print("Edge shape:", edge.shape)

# # Fusion weight
# alpha = 0.6  # you will experiment later

# # Fuse
# fused = alpha * edge + (1 - alpha) * raw

# # Save
# np.save("features/fused_embeddings.npy", fused)

alphas = [0.3, 0.5, 0.7, 0.9]

for alpha in alphas:
    fused = alpha * edge + (1 - alpha) * raw
    np.save(f"features/fused_{alpha}.npy", fused)

print("All fusion versions saved!")

print("Fused embeddings created!")
print("Fused shape:", fused.shape)