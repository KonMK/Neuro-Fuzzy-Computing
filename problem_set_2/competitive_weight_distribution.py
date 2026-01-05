import numpy as np
import matplotlib.pyplot as plt

# Initial Weights W(0)
W_initial = np.array([
    [0.0, 0.0],   # w1(0)
    [1.0, 0.0],   # w2(0)
    [1.0, 1.0],   # w3(0)
    [0.0, -1.0]   # w4(0)
])

# Final Weights W(1)
W_final = np.array([
    [-0.5, 0.5],  # w1(1)
    [0.0, 0.5],   # w2(1)
    [0.0, 1.0],   # w3(1)
    [0.0, -1.0]   # w4(1)
])

# Input Vector p
p = np.array([-1.0, 1.0])

plt.figure(figsize=(8, 8))
plt.scatter(W_initial[:, 0], W_initial[:, 1], 
            marker='o', color='blue', s=100, label='Initial Weights $w_i(0)$')
plt.scatter(W_final[:, 0], W_final[:, 1], 
            marker='x', color='red', s=100, label='Final Weights $w_i(1)$')
plt.scatter(p[0], p[1], 
            marker='*', color='green', s=300, label='Input Vector $p$')
plt.text(p[0] + 0.1, p[1] + 0.1, '$p$', fontsize=12, color='green')


for i in range(len(W_initial)):
    dx = W_final[i, 0] - W_initial[i, 0]
    dy = W_final[i, 1] - W_initial[i, 1]
    
    plt.arrow(W_initial[i, 0], W_initial[i, 1], dx, dy, 
              head_width=0.05, head_length=0.05, fc='gray', ec='gray', 
              linewidth=0.8, linestyle='--', zorder=0)
    
    plt.text(W_initial[i, 0] - 0.1, W_initial[i, 1] - 0.15, f'$w_{i+1}$ (0)', fontsize=10, color='blue')
    
    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        plt.text(W_final[i, 0] + 0.05, W_final[i, 1] + 0.05, f'$w_{i+1}$ (1)', fontsize=10, color='red')
    else:
        plt.text(W_final[i, 0] - 0.1, W_final[i, 1] - 0.15, f'$w_{i+1}$ (1)', fontsize=10, color='red')

plt.title('Movement of Weight Vectors after One Iteration')
plt.xlabel('$w_1$ component')
plt.ylabel('$w_2$ component')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.axis('equal') 
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)

plot_filename = 'problem13.pdf'
plt.savefig(plot_filename)
