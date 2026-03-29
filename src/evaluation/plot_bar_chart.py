import matplotlib.pyplot as plt
import numpy as np

methods = ["Raw", "Edge", "F(0.3)", "F(0.5)", "F(0.7)", "F(0.9)"]
p5 = [0.285, 0.117, 0.294, 0.290, 0.290, 0.184]

x = np.arange(len(methods))

plt.figure()

plt.bar(x, p5)

plt.xticks(x, methods)
plt.xlabel("Method")
plt.ylabel("Precision@5")
plt.title("Comparison of Methods (P@5)")
plt.grid(axis='y')

plt.savefig("reports/figures/bar_p5.png")
plt.show()