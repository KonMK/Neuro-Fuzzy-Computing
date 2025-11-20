import numpy as np
import matplotlib.pyplot as plt

p1 = np.array([1, 2])
t1 = -1
p2 = np.array([-2, 1])
t2 = 1
P = np.column_stack((p1, p2))
T = np.array([t1, t2])

# Compute optimal weights
w_opt = np.linalg.inv(P @ P.T) @ (P @ T)
print("Optimal weights:", w_opt)

# Define the Mean Squared Error surface
def mse_surface(w1, w2):
    w = np.array([w1, w2])
    e1 = t1 - np.dot(w, p1)
    e2 = t2 - np.dot(w, p2)
    J = 0.25 * (e1**2 + e2**2)
    return J

# Create grid for contour plot
w1_vals = np.linspace(-2, 2, 200)
w2_vals = np.linspace(-2, 2, 200)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
J_vals = np.array([[mse_surface(w1, w2) for w1, w2 in zip(row1, row2)] 
                   for row1, row2 in zip(W1, W2)])

# LMS Simulation
alpha = 0.05  # learning rate (small)
w = np.array([0.0, 1.0])  # initial weights
trajectory = [w.copy()]

# Perform several epochs of LMS
for epoch in range(30):
    for p, t in zip([p1, p2], [t1, t2]):
        a = np.dot(w, p)
        e = t - a
        w = w + alpha * e * p
        trajectory.append(w.copy())

trajectory = np.array(trajectory)


# Plot the MSE contours and LMS trajectory
plt.figure(figsize=(8, 6))
contours = plt.contour(W1, W2, J_vals, levels=20, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(w_opt[0], w_opt[1], 'r*', markersize=12, label='Optimal weights')
plt.plot(trajectory[:,0], trajectory[:,1], 'bo-', label='LMS trajectory')
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title('Contour Plot of MSE with LMS Trajectory')
plt.legend()
plt.grid(True)
plt.show()

# Plot Decision Boundary
# x2 = -3*x1
x1 = np.linspace(-3, 3, 100)
x2 = -3 * x1

plt.figure(figsize=(6, 6))
plt.plot(x1, x2, 'r-', label='Decision Boundary')
plt.plot(p1[0], p1[1], 'bo', label='Class -1')
plt.plot(p2[0], p2[1], 'go', label='Class +1')
plt.xlabel('$p_1$')
plt.ylabel('$p_2$')
plt.title('Optimal Decision Boundary')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

