import matplotlib.pyplot as plt

methods = ["Raw", "Edge", "F(0.3)", "F(0.5)", "F(0.7)", "F(0.9)"]

p3 = [0.407, 0.123, 0.400, 0.397, 0.397, 0.220]
p5 = [0.285, 0.117, 0.294, 0.290, 0.290, 0.184]
p10 = [0.198, 0.117, 0.200, 0.205, 0.196, 0.153]

x = range(len(methods))

plt.figure()

plt.plot(x, p3, marker='o', label="P@3")
plt.plot(x, p5, marker='o', label="P@5")
plt.plot(x, p10, marker='o', label="P@10")

plt.xticks(x, methods)
plt.xlabel("Method")
plt.ylabel("Precision")
plt.title("Precision@K Comparison Across Methods")
plt.legend()
plt.grid(True)

plt.savefig("reports/figures/precision_comparison.png")
plt.show()