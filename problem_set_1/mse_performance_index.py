import numpy as np
import matplotlib.pyplot as plt

# Mean Square Error (MSE) function F(w1, w2)
def mse_performance_index(w1, w2):
    return(0.8 + 0.1*w1 - 1.5*w2 + 0.825*(w1**2) - 0.25*w1*w2 + 1.2*(w2**2))

# Optimal weight vector w* (w1_star, w2_star)
w1_star = 0.0346
w2_star = 0.6491
F_min = mse_performance_index(w1_star, w2_star)

# Grid of w1 and w2 values for the contour plot
w1_range = np.linspace(w1_star - 2, w1_star + 2, 100)
w2_range = np.linspace(w2_star - 1.5, w2_star + 1.5, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)

F = mse_performance_index(W1, W2)
plt.figure(figsize=(8, 6))

# Custom levels to better show the minimum
custom_levels = [F_min + 0.005, F_min + 0.01, F_min + 0.025, F_min + 0.05, F_min + 0.1, F_min + 0.2, F_min + 0.4, F_min + 0.8]
custom_levels = [l for l in custom_levels if l < np.max(F)] # Discard levels above max F

CS = plt.contour(W1, W2, F, levels=custom_levels, colors='blue', linewidths=0.8)
plt.clabel(CS, inline=1, fontsize=8, fmt='%1.3f')

plt.plot(w1_star, w2_star, 'r*', markersize=12, label=fr'Optimal Weight $\mathbf{{w}}^*$ ({w1_star:.4f}, {w2_star:.4f})')

plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title(r'Contour Plot of Mean Square Error Performance Index $F(\mathbf{w})$')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig('Problem6.pdf', format='pdf')
