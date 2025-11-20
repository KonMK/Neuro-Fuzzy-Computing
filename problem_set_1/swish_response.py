import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def swish(x):
    "Swish activation"
    return x / (1 + np.exp(-x))

def leaky_relu(x, alpha=0.1):
    """Leaky ReLU activation"""
    return np.where(x >= 0, x, alpha * x)

# Define parameters
# Layer 1
w11_1 = -0.27
w21_1 = -0.41
b1_1 = -0.48
b2_1 = -0.13

# Layer 2
w11_2 = 0.09
w12_2 = -0.17
b1_2 = 0.48

# Input range
p = np.linspace(-2, 2, 400)

# Compute layer 1
n1_1 = w11_1 * p + b1_1
a1_1 = swish(n1_1)

n2_1 = w21_1 * p + b2_1
a2_1 = swish(n2_1)

#  Compute layer 2
n2 = w11_2 * a1_1 + w12_2 * a2_1 + b1_2
a2 = leaky_relu(n2)

# Plot results
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(p, n1_1)
plt.title("i. n_1^1 = w_{1,1}^1·p + b_1^1")
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(p, a1_1)
plt.title("ii. a_1^1 = Swish(n_1^1)")
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(p, n2_1)
plt.title("iii. n_2^1 = w{2,1}^1·p + b_2^1")
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(p, a2_1)
plt.title("iv.a_2^1 = Swish(n_2^1)")
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(p, n2)
plt.title("v. n_1^2 = w{1,1}^2·a_1^1 + w{1,2}^2·a_2^1 + b_1^2")
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(p, a2)
plt.title("vi. a_1^2 = LeakyReLU(n_1^2)")
plt.grid(True)

plt.tight_layout()
plt.show()
