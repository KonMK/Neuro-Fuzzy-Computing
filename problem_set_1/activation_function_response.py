import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def logsig(n:list[np.ndarray]) -> list[np.ndarray]:
    return 1/(1 + np.power(np.e, n))

def ReLU(n:list[np.ndarray]) -> list[np.ndarray]:
    n[n>0] = 0
    return n

def net_input(p:list[np.ndarray], w:float, b:float) -> list[np.ndarray]:
    return w*p + b

def main():
    func_used = ["logsig", "ReLU"]
    w = [[-2.0, 1.5], [-1.0, 2.5]]; b = [-1.0, -0.2, -2.0]
    p = np.linspace(-2, 2, 500)

    fig = plt.figure(figsize=(10, 8))
    gs_outer = gridspec.GridSpec(2, 1, hspace=0.5)
    gs_inner = [gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[i], wspace=0.2, hspace=0.5) for i in range(2)]

    for i in range(2):
        a_inner = [logsig(net_input(p, w[index][0], b[index])) for index in range(2)] if i==0 else [ReLU(net_input(p, w[index][0], b[index])) for index in range(2)]
        a = w[1][0]*a_inner[0] + w[1][1]*a_inner[1] + b[2]

        fig.text(0.5, 0.94 - i*0.46, func_used[i], ha='center', va='center', fontsize=16, fontweight='bold')

        ax1 = plt.Subplot(fig, gs_inner[i][0, 0])
        ax1.plot(p, a_inner[0])
        ax1.set_title("a1 graph")
        ax1.set_xlabel("p")
        ax1.set_ylabel("a1")
        fig.add_subplot(ax1)

        ax2 = plt.Subplot(fig, gs_inner[i][0, 1])
        ax2.plot(p, a_inner[1])
        ax2.set_title("a2 graph")
        ax2.set_xlabel("p")
        ax2.set_ylabel("a2")
        fig.add_subplot(ax2)

        ax3 = plt.Subplot(fig, gs_inner[i][1, :])
        ax3.plot(p, a)
        ax3.set_title("a graph")
        ax3.set_xlabel("p")
        ax3.set_ylabel("a")
        fig.add_subplot(ax3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
