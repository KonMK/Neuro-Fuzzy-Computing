import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def generate_ar_data(n_samples):
    a = [0.5, -0.25, 0.1, -0.2]
    X = np.zeros(n_samples + 4)
    # Uniform noise in (0, 0.05)
    U = np.random.uniform(0, 0.05, n_samples + 4)
    
    for t in range(4, n_samples + 4):
        X[t] = a[0]*X[t-1] + a[1]*X[t-2] + a[2]*X[t-3] + a[3]*X[t-4] + U[t]
    return X[4:]

def create_dataset(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

sample_sizes = [200, 500, 1000, 2500, 5000]
mse_results = []

for size in sample_sizes:
    # Generate and scale data
    raw_data = generate_ar_data(size + 100) # extra for windowing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(raw_data.reshape(-1, 1)).flatten()
    
    win_size = 10
    X_train, y_train = create_dataset(scaled_data, win_size)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    # LSTM Model
    model = Sequential([
        LSTM(32, input_shape=(win_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Training
    print(f"Training with {size} samples...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    mse_results.append(history.history['loss'][-1])

# Accuracy (Loss) vs Training Samples
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, mse_results, marker='o', linestyle='-', color='b')
plt.title('RNN Prediction Error vs. Training Sample Size')
plt.xlabel('Number of Training Samples')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.savefig('problem2.pdf')
