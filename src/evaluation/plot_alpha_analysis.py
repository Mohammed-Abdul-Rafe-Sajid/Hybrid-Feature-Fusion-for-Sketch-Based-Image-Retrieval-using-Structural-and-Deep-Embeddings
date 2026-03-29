import matplotlib.pyplot as plt

alphas = [0.0, 0.3, 0.5, 0.7, 0.9]

p3 = [0.407, 0.400, 0.397, 0.397, 0.220]
p5 = [0.285, 0.294, 0.290, 0.290, 0.184]
p10 = [0.198, 0.200, 0.205, 0.196, 0.153]

plt.figure()

plt.plot(alphas, p3, marker='o', label="P@3")
plt.plot(alphas, p5, marker='o', label="P@5")
plt.plot(alphas, p10, marker='o', label="P@10")

plt.xlabel("Alpha (Edge Weight)")
plt.ylabel("Precision")
plt.title("Effect of Fusion Weight on Retrieval Performance")
plt.legend()
plt.grid(True)

plt.savefig("reports/figures/alpha_analysis.png")
plt.show()