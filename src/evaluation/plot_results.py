import matplotlib.pyplot as plt

# Data from your results
alphas = [0, 0.3, 0.5, 0.7, 0.9]

# P@5 values (update if needed)
p5_scores = [
    0.285,  # Raw (α=0)
    0.294,
    0.290,
    0.290,
    0.184
]

plt.figure()

plt.plot(alphas, p5_scores, marker='o')
plt.xlabel("Alpha (Edge Weight)")
plt.ylabel("Precision@5")
plt.title("Effect of Feature Fusion on Retrieval Performance")

plt.grid(True)

plt.savefig("reports/figures/fusion_graph.png")
plt.show()