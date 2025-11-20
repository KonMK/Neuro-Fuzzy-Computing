from itertools import chain
from random import choice
import numpy as np
import matplotlib.pyplot as plt

INIT_WEIGHTS = 0.0
LEARNING_RATE = 0.03
TRAIN_STEPS = 100

def net_output(p:np.ndarray[np.float64], w:np.ndarray[np.float64], w_b:float) -> np.float64:
    return np.sum(w*p) + w_b # w_b*1

def genPatterns(pattern:np.ndarray) -> list[np.ndarray[np.float64]]:
    moveable_p = [-1.0, -1.0, -1.0, -1.0]
    return [np.insert(pattern, 0, moveable_p) if i == 0 else np.insert(pattern, len(pattern), moveable_p) for i in range(2)]

def main():
    patterns = np.array([
        [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0], # T
        [1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0],     # G
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0]   # F
    ], dtype=np.float64)
    p = list(chain.from_iterable(genPatterns(pattern) for pattern in patterns))
    weights = np.array([INIT_WEIGHTS]*16, dtype=np.float64); w_b = 0.0
    target = [60, 0, -60]

    sse_list:list = []
    indexes = list(range(6))
    for i in range(TRAIN_STEPS):
        if (i%6 == 0):
            picks:list = indexes.copy()

        index = choice(picks)
        picks.remove(index)

        error = target[int(index/2)] - net_output(p[index], weights, w_b)
        mul = LEARNING_RATE*error
        weights += mul*p[index]
        w_b += mul # *1

        total_sse = 0
        for j in range(6):
            error_i = target[int(j/2)] - net_output(p[j], weights, w_b)
            total_sse += error_i * error_i
        sse_list.append(total_sse)

    _, ax = plt.subplots(1)
    ax.plot([i for i in range(TRAIN_STEPS)], sse_list)
    ax.set_title("SUM SQUARE ERROR VERSUS TRAINING STEPS")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
