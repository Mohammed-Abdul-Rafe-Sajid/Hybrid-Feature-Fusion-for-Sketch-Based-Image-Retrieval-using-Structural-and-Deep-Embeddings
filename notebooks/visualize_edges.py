import cv2
import matplotlib.pyplot as plt
import os

raw_path = "data/sample/raw"
edge_path = "data/sample/edges"

files = os.listdir(raw_path)[:5]

plt.figure(figsize=(10,4))

for i, file in enumerate(files):
    # Original
    img = cv2.imread(os.path.join(raw_path, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Edge
    edge = cv2.imread(os.path.join(edge_path, file), cv2.IMREAD_GRAYSCALE)

    plt.subplot(2,5,i+1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2,5,i+6)
    plt.imshow(edge, cmap="gray")
    plt.title("Edge")
    plt.axis("off")

plt.savefig("reports/figures/edge_vs_original.png")
plt.show()