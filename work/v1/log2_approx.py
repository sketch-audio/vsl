import numpy as np
import approx_helpers as approx

f = np.log2
order = 8
lo = 1
hi = 2

m_coeffs = approx.calc_minimax(f, order, lo, hi, a=1)

x = np.linspace(lo, hi, 1000)
y = f(x)
y_approx = approx.eval_poly(x, m_coeffs, a=1)

# Remove where y is close to zero
mask = np.abs(y) > 1e-9
x = x[mask]
y = y[mask]
y_approx = y_approx[mask]

print(f"Log2 approximation, order {order} on [{lo}, {hi}]")
approx.print_coeffs(m_coeffs, "Minimax")
approx.print_max_abs_error(x, y, y_approx)
approx.print_max_rel_error(x, y, y_approx)

approx.plot_approx(x, y, y_approx, 'log2(x)', lo, hi)