from fractions import Fraction
import numpy as np

def to_dec_list(lst):
    return [float(v) for v in lst]

def fmt_num(n, prec=6):
    return f"{float(n):.{prec}f}"

# Define the diagonal Hessian matrix A as fractions
A = [Fraction(2), Fraction(10), Fraction(6)]
x0 = [Fraction(2), Fraction(2), Fraction(2)]

# Initialize
x = x0.copy()
r = [-A[i] * x[i] for i in range(3)]  # Residual: r0 = -A x0
p = r.copy()

print("Step 0:")
print(f"  x0 = {to_dec_list(x)}")
print(f"  r0 = {to_dec_list(r)}")
print(f"  p0 = {to_dec_list(p)}\n")

max_iter = 3

for k in range(max_iter):
    # Compute alpha = (r^T r) / (p^T A p)
    rTr = sum([ri**2 for ri in r])
    pTAp = sum([p[i] * A[i] * p[i] for i in range(3)])
    alpha = rTr / pTAp

    # Update x
    x_new = [x[i] + alpha * p[i] for i in range(3)]
    # Update residual
    r_new = [r[i] - alpha * A[i] * p[i] for i in range(3)]

    
    # Print analytical step
    print(f"Step {k+1}:")
    print(f"  alpha = {fmt_num(alpha)}")
    print(f"  x = {to_dec_list(x_new)}")
    print(f"  r = {to_dec_list(r_new)}")

    # Compute beta (if not last iteration)
    if k < max_iter - 1:
        rTr_new = sum([ri**2 for ri in r_new])
        beta = rTr_new / rTr
        p = [r_new[i] + beta * p[i] for i in range(3)]
        print(f"  beta = {fmt_num(beta)}")
        print(f"  p = {to_dec_list(p)}\n")

    # Prepare for next iteration
    x = x_new
    r = r_new

print(f"\nFinal solution: x* = {to_dec_list(x)}")
