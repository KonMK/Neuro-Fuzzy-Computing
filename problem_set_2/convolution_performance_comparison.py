import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt


# Load image and convert to grayscale
img = Image.open("cat_grayscale.bmp").convert("L")
I = np.asarray(img, dtype=np.float64)

# Define filters
F1 = np.array([[0,  -1,  0],
               [-1, 8,  -1],
               [0,  -1,  0]])

F2 = np.array([[0, 1, 0],
               [1, 4, 1],
               [0, 1, 0]])

F3 = np.array([[-1, -1, -1],
               [-1,8, -1],
               [-1, -1, -1]])


# Direct valid convolution
def conv2d_valid(I, F):
    h, w = I.shape
    fh, fw = F.shape
    out = np.zeros((h - fh + 1, w - fw + 1))

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(I[i:i+fh, j:j+fw] * F)

    return out


# im2col (Toeplitz)
def im2col(I, fh, fw):
    h, w = I.shape
    cols = []
    for i in range(h - fh + 1):
        for j in range(w - fw + 1):
            cols.append(I[i:i+fh, j:j+fw].ravel())
    return np.asarray(cols)

def conv2d_matmul(I, F):
    fh, fw = F.shape
    X = im2col(I, fh, fw)
    y = X @ F.ravel()
    return y.reshape(I.shape[0]-fh+1, I.shape[1]-fw+1)

# Timing direct convolutions
t0 = time.time()
C1 = conv2d_valid(I, F1)
t1 = time.time()
C2 = conv2d_valid(I, F2)
t2 = time.time()
C3 = conv2d_valid(I, F3)
t3 = time.time()

print("Direct convolution times:")
print("C1:", t1 - t0)
print("C2:", t2 - t1)
print("C3:", t3 - t2)


# Timing matrix multiplication convolutions
t0 = time.time()
C1m = conv2d_matmul(I, F1)
t1 = time.time()
C2m = conv2d_matmul(I, F2)
t2 = time.time()
C3m = conv2d_matmul(I, F3)
t3 = time.time()

print("\nMatrix multiplication times:")
print("C1:", t1 - t0)
print("C2:", t2 - t1)
print("C3:", t3 - t2)


# Display results
plt.figure(figsize=(12,4))

plt.subplot(131)
plt.imshow(C1, cmap="gray")
plt.title("C1")
plt.axis("off")

plt.subplot(132)
plt.imshow(C2, cmap="gray")
plt.title("C2")
plt.axis("off")

plt.subplot(133)
plt.imshow(C3, cmap="gray")
plt.title("C3")
plt.axis("off")

plt.show()
