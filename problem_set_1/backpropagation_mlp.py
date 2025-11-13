import numpy as np
import matplotlib.pyplot as plt

# --- Activation Functions ---

def logsig(n):
    # Log-sigmoid activation function: a = 1 / (1 + exp(-n))
    n_clipped = np.clip(n, -500, 500)
    return 1.0 / (1.0 + np.exp(-n_clipped))

def dlogsig_da(a):
    # Derivative of log-sigmoid w.r.t. net input, expressed using output a: F'(n) = a * (1 - a)
    return a * (1.0 - a)

# The linear activation derivative (F'(n) = 1) is simply the scalar 1.

# --- Target Function ---

def g(p):
    # Target function: g(p) = 1 + sin(p * pi/3)
    return 1.0 + np.sin(p * (np.pi / 3.0))

# --- Weight Initialization ---

def initialize_weights(S1):
    # Initializes weights and biases uniformly between -0.5 and 0.5.
    # W1: S1 x 1 matrix
    W1 = np.random.uniform(-0.5, 0.5, (S1, 1))
    # b1: S1 x 1 matrix (bias vector)
    b1 = np.random.uniform(-0.5, 0.5, (S1, 1))

    # W2: 1 x S1 matrix
    W2 = np.random.uniform(-0.5, 0.5, (1, S1))
    # b2: 1 x 1 matrix (scalar bias)
    b2 = np.random.uniform(-0.5, 0.5, (1, 1))

    return W1, b1, W2, b2

# --- Backpropagation Training Function (Online/Sequential Learning) ---

def train_network(S1, alpha, epochs=10000, initial_condition=1):
    np.random.seed(initial_condition)
    print(f"\n--- Training S1={S1}, alpha={alpha}, Seed={initial_condition} ---")

    # Training data (P must be a column vector for matrix operations)
    P_range = np.linspace(-2, 2, 50) # 50 points in [-2, 2]
    T_data = g(P_range)

    # Weights and biases
    W1, b1, W2, b2 = initialize_weights(S1)

    error_history = []
    
    # Maximum magnitude for weights/biases to prevent overflow 
    MAX_WEIGHT_MAGNITUDE = 10.0 
    # Placeholder error for indicating failure
    DIVERGED_ERROR = 1000.0

    for epoch in range(epochs):
        epoch_mse = 0.0
        
        # Each pattern (p) in the training set
        for i in range(len(P_range)):
            # p_i: current input pattern (1x1 column vector)
            p_i = np.array([[P_range[i]]])
            # t_i: current target (1x1 column vector)
            t_i = np.array([[T_data[i]]])

            # --- FORWARD PASS (Matrix Operations) ---
            
            # Layer 1 (Hidden)
            n1 = W1 @ p_i + b1  # (S1x1) = (S1x1) @ (1x1) + (S1x1)
            a1 = logsig(n1)     # (S1x1)

            # Layer 2 (Output, Linear)
            n2 = W2 @ a1 + b2  # (1x1) = (1xS1) @ (S1x1) + (1x1)
            a2 = n2            # (1x1), linear activation

            # --- BACKWARD PASS (Matrix Operations) ---
            
            # Output Layer (Layer 2) Error and Sensitivity
            e = t_i - a2
            
            # Sensitivity delta2: F'(n2) * e. F' for linear is 1.
            delta2 = e # (1x1)

            # Update W2 and b2
            # dW2 = alpha * delta2 * (a1)^T
            W2 = W2 + alpha * delta2 @ a1.T
            # db2 = alpha * delta2
            b2 = b2 + alpha * delta2

            # Hidden Layer (Layer 1) Sensitivity
            # delta1: F'(n1) * (W2)^T * delta2
            
            # F'(n1) = diag(a1 * (1 - a1)). Element-wise
            F1_prime = dlogsig_da(a1) # (S1x1)

            # (W2)^T @ delta2: (S1x1) = (S1x1) @ (1x1)
            backprop_error = W2.T @ delta2 
            
            # delta1 = F1_prime * backprop_error (Element-wise multiplication)
            delta1 = F1_prime * backprop_error # (S1x1)

            # Update W1 and b1s
            # dW1 = alpha * delta1 * (p_i)^T
            W1 = W1 + alpha * delta1 @ p_i.T
            # db1 = alpha * delta1
            b1 = b1 + alpha * delta1
                        
            if np.isnan(W1).any() or np.isinf(W1).any() or \
               np.isnan(W2).any() or np.isinf(W2).any():
                print(f"Divergence at Epoch {epoch+1}: Weights became NaN/Inf. Training Halted.")
                return DIVERGED_ERROR # Return to signal failure

            # Weights and biases clipping to prevent runaway growth
            W1 = np.clip(W1, -MAX_WEIGHT_MAGNITUDE, MAX_WEIGHT_MAGNITUDE)
            b1 = np.clip(b1, -MAX_WEIGHT_MAGNITUDE, MAX_WEIGHT_MAGNITUDE)
            W2 = np.clip(W2, -MAX_WEIGHT_MAGNITUDE, MAX_WEIGHT_MAGNITUDE)
            b2 = np.clip(b2, -MAX_WEIGHT_MAGNITUDE, MAX_WEIGHT_MAGNITUDE)
            
            epoch_mse += np.sum(e**2) 

        # Mean Squared Error value for the epoch
        current_mse = epoch_mse / len(P_range)
        error_history.append(current_mse)
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs} | MSE: {current_mse:.6f}")

    print(f"Final MSE: {error_history[-1]:.6f}\n")

    # --- Final Approximation and Plot ---
    # WARNING: USED FOR DEBUGGING/VISUAL REFERENCE ONLY. 
    # IF UNCOMMENTED, USE COMMENTED def run_single() TO EXECUTE BY SINGLE ITERATIONS 
    # AND NOT THE FULL LOOP, OTHERWISE IT MAY CRASH YOUR PC [has happened]

    # # Dense set of test points
    # P_test = np.linspace(-3, 3, 200)
    # A2_approx = []
    
    # # Test network with final weights (Forward Pass only)
    # for p_val in P_test:
    #     p_test_i = np.array([[p_val]])
        
    #     n1 = W1 @ p_test_i + b1
    #     a1 = logsig(n1)
        
    #     n2 = W2 @ a1 + b2
    #     a2 = n2
    #     A2_approx.append(a2[0, 0])
    
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # fig.suptitle(f'Approximation of $g(p)$ with $S_1={S1}$, $\\alpha={alpha}$, Seed={initial_condition}', fontsize=14)

    # # Subplot 1: Function Approximation
    # ax1.plot(P_test, g(P_test), 'k--', label='Target Function $g(p)$')
    # ax1.plot(P_test, A2_approx, 'r-', label='MLP Approximation')
    # ax1.scatter(P_range, T_data, s=5, c='b', label='Training Points')
    # ax1.set_title('Function Approximation')
    # ax1.set_xlabel('$p$')
    # ax1.set_ylabel('$g(p)$')
    # ax1.set_xlim([-3, 3])
    # ax1.grid(True)
    # ax1.legend()
    
    # # Subplot 2: Training Error
    # ax2.plot(error_history, 'b-')
    # ax2.set_title('Training MSE Convergence')
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Mean Squared Error (MSE)')
    # ax2.set_yscale('log')
    # ax2.grid(True)

    # plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    # plt.savefig('plot.pdf', format='pdf')
    
    return error_history[-1] # Return final MSE

# --- Experiment Runners ---

def run_single(S1=2, alpha=0.01, seed=17, epochs=10000):
    DIVERGED_ERROR = 1000.0
    
    print(f"\n--- Running Single Case: S1={S1}, alpha={alpha}, Seed={seed}, Epochs={epochs} ---")
    final_mse = train_network(S1, alpha, epochs=epochs, initial_condition=seed)
    status = "SUCCESS" if final_mse < DIVERGED_ERROR else "DIVERGED"
    print(f"Final MSE: {final_mse:.6f} ({status})")

def run_experiments():
    DIVERGED_ERROR = 1000.0

    S1_values = [2, 6, 10, 20]
    alpha_values = [0.01, 0.1, 0.2, 0.5]
    initial_conditions = [17, 42, 101, 202] # Random seeds for initial weights

    results = {}
    
    for S1 in S1_values:
        for alpha in alpha_values:
            for seed in initial_conditions:
                key = (S1, alpha, seed)
                final_mse = train_network(S1, alpha, epochs=10000, initial_condition=seed)
                results[key] = final_mse
    
    print("\n\n-------------------------------------------------------")
    print("           EXPERIMENT SUMMARY (Final MSE)              ")
    print("-------------------------------------------------------")
    
    for (S1, alpha, seed), mse in results.items():
        status = " (DIVERGED/FAILED)" if mse >= DIVERGED_ERROR else ""
        print(f"S1={S1}, alpha={alpha:.2f}, Seed={seed}: MSE = {mse:.6f}{status}")
    print("-------------------------------------------------------")
    
def main():
    # Full experiment and consolidated presentation for statistical interpretation
    run_experiments()
    
    # Uncomment for a single iteration to produce the plot 
    # Accepts values beyong the defaults
    # run_single()
    
if __name__ == "__main__":
    main()
    
