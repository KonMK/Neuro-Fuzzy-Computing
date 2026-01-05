import numpy as np

def radbas(n:float):
    return pow(np.e, -pow(n, 2))

def hidden_layer(p:float, w:float, b:float):
    return (p - w), abs(p - w)*b

def lin_layer(a:float, w:float, b:float):
    return a*w + b

def main():
    w1 = 0.0; b1 = 1.0
    w2 = -2.0; b2 = 1.0
    alpha = 1.0
    tar_set = [(-1, 0), (1, 1)]

    for i in range(2):
        r_vals = []
        n_vals = []
        a_vals = []
        y_vals = []
        e_vals = []
    
        for p, t in tar_set:
            r, n = hidden_layer(p, w1, b1)
            a = radbas(n)
            y = lin_layer(a, w2, b2)

            r_vals.append(r)
            n_vals.append(n)
            a_vals.append(a)
            y_vals.append(y)
            e_vals.append(y - t)
        
        dE_dw1 = 0.0
        dE_db1 = 0.0
        dE_dw2 = 0.0
        dE_db2 = 0.0
        for k in range(2):
            dE_dw1 += e_vals[k] * w2 *  2 * r_vals[k] * (b1**2) * a_vals[k]
            dE_db1 += e_vals[k] * w2 * -2 * b1 * (r_vals[k]**2) * a_vals[k]
            dE_dw2 += e_vals[k] * a_vals[k]
            dE_db2 += e_vals[k]

        w1 = w1 - alpha * dE_dw1
        b1 = b1 - alpha * dE_db1
        w2 = w2 - alpha * dE_dw2
        b2 = b2 - alpha * dE_db2
        
        print(f"\nIteration {i+1}\n===================")
        print(f"p={p}, t={t}, y={y:.4f}")
        print(f"w1={w1:.4f}, b1={b1:.4f}, w2={w2:.4f}, b2={b2:.4f}")

if __name__ == "__main__":
    main()
