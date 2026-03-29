import cv2
import matplotlib.pyplot as plt
import os

path = "data/sample/raw"

files = os.listdir(path)[:5]

plt.figure(figsize=(10,2))

for i, f in enumerate(files):
    img = cv2.imread(os.path.join(path, f))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.axis("off")

plt.savefig("reports/figures/stl10_samples.png")
plt.show()