import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
START_POINT = np.array([8.0, 8.0]) # Initial starting point
ITERATIONS = 500
RHO = 0.9      # Decay rate for Adadelta
EPSILON = 1e-6 # Stability term

# --- Helper Functions ---

def func_aligned(w):
    return 0.1 * w[0]**2 + 2 * w[1]**2

def grad_aligned(w):
    grad_w1 = 0.2 * w[0]
    grad_w2 = 4.0 * w[1]
    return np.array([grad_w1, grad_w2])

def func_rotated(w):
    term1 = w[0] + w[1]
    term2 = w[0] - w[1]
    return 0.1 * (term1**2) + 2 * (term2**2)

def grad_rotated(w):
    term1 = w[0] + w[1]
    term2 = w[0] - w[1]
    # dF/dw1 = 0.2(w1+w2) + 4(w1-w2)
    grad_w1 = 0.2 * term1 + 4.0 * term2
    # dF/dw2 = 0.2(w1+w2) - 4(w1-w2)
    grad_w2 = 0.2 * term1 - 4.0 * term2
    return np.array([grad_w1, grad_w2])

def adadelta_optimizer(grad_func, x0, learning_rate, iterations=100, rho=0.9, eps=1e-6):
    w = x0.copy()
    trajectory = [w.copy()]
    
    # Averages
    # s_t = E[g^2]
    eg2 = np.zeros_like(w)
    # delta_w = E[dx^2]
    edx2 = np.zeros_like(w)
    
    for _ in range(iterations):
        g = grad_func(w)
        
        # 1. Accumulate gradient
        eg2 = rho * eg2 + (1 - rho) * (g**2)
        
        # 2. Compute update (RMS(dx) / RMS(g)) * g
        rms_g = np.sqrt(eg2 + eps)
        rms_dx = np.sqrt(edx2 + eps)
        
        delta_w = - (rms_dx / rms_g) * g
        
        # 3. Apply update with chosen learning rates
        w_new = w + learning_rate * delta_w
        
        # 4. Accumulate updates
        edx2 = rho * edx2 + (1 - rho) * (delta_w**2)
        
        w = w_new
        trajectory.append(w.copy())
        
    return np.array(trajectory)

def plot_trajectory(func, trajectory, title, filename):
    x_vals = np.linspace(-10, 10, 100)
    y_vals = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
            
    plt.figure(figsize=(8, 6))
    
    # Using log spacing for levels to see details near the minimum
    levels = np.logspace(-2, 3, 20)
    cp = plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    plt.colorbar(cp, label='F(w)')
    
    # Plots
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', label='Adadelta Trajectory', markersize=3, linewidth=1)
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'ko', label='Start')
    plt.plot(0, 0, 'b*', markersize=12, label='Global Min')
    
    plt.title(title)
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()
    

if __name__ == "__main__":
    # Aligned function, alpha=0.4
    traj1 = adadelta_optimizer(grad_aligned, START_POINT, learning_rate=0.4, iterations=ITERATIONS)
    plot_trajectory(func_aligned, traj1, 
                   'F(w) Aligned, LR=0.4', 
                   'problem05_1.pdf')

    # Aligned function, alpha=3.0
    traj2 = adadelta_optimizer(grad_aligned, START_POINT, learning_rate=3.0, iterations=ITERATIONS)
    plot_trajectory(func_aligned, traj2, 
                   'F(w) Aligned, LR=3.0', 
                   'problem05_2.pdf')

    # Rotated function, alpha=1.0 (standard alpha chosen)
    traj3 = adadelta_optimizer(grad_rotated, START_POINT, learning_rate=1.0, iterations=ITERATIONS)
    plot_trajectory(func_rotated, traj3, 
                   'F(w) Rotated, LR=1.0', 
                   'problem05_3.pdf')
