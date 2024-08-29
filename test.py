import pysr
import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

# Sample dataset
np.random.seed(0)
N = 3000
upper_sigma = 5
X = 2 * np.random.rand(N, 5)
sigma = np.random.rand(N) * (5 - 0.1) + 0.1
eps = sigma * np.random.randn(N)
y = 5 * np.cos(3.5 * X[:, 0]) - 1.3 + eps

plt.scatter(X[:, 0], y, alpha=0.2)
plt.xlabel("x_0")
plt.ylabel("y")

weights = 1 / sigma**2

# Learn equations
model = PySRRegressor(
    extra_sympy_mappings={"myloss": lambda x, y, w: w * sympy.Abs(x - y)},  # Custom loss function with weights.
    niterations=20,
    populations=20,  # Use more populations
    binary_operators=["+", "*"],
    unary_operators=["cos"],
)
model.fit(X, y, weights=weights)

# Extract the best equation
best_idx = model.equations_.query(
    f"loss < {2 * model.equations_.loss.min()}"
).score.idxmax()
best_eq = model.sympy(best_idx)
print("Best Equation:", best_eq)

# Generate predictions
y_prediction = model.predict(X, index=best_idx)

# Plot the results
plt.scatter(X[:, 0], y, alpha=0.2, label="Original Data")
plt.scatter(X[:, 0], y_prediction, alpha=0.2, label="Predicted Data", color='r')
plt.xlabel("x_0")
plt.ylabel("y")
plt.legend()
plt.show()
