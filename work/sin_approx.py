import numpy as np
import approx_helpers as approx

f = np.sin
order = 11
lo = -np.pi
hi = np.pi

m_coeffs = approx.calc_minimax(f, order, lo, hi)

x = np.linspace(lo, hi, 1000)
y = f(x)
y_approx = approx.eval_poly(x, m_coeffs)

print(f"Sine approximation, order {order} on [{lo}, {hi}]")
approx.print_coeffs(m_coeffs, "Minimax")
approx.print_max_abs_error(x, y, y_approx)
approx.print_max_rel_error(x, y, y_approx)

approx.plot_approx(x, y, y_approx, 'sin(x)', lo, hi)

# sin(2πx)
f = lambda x: np.sin(2 * np.pi * x)
order = 9

lo = -0.5
hi = 0.5

m_coeffs = approx.calc_minimax(f, order, lo, hi)

x = np.linspace(lo, hi, 1000)
y = f(x)
y_approx = approx.eval_poly(x, m_coeffs)

print(f"\nSine(2πx) approximation, order {order} on [{lo}, {hi}]")
approx.print_coeffs(m_coeffs, "Minimax")
approx.print_max_abs_error(x, y, y_approx)
approx.print_max_rel_error(x, y, y_approx)

approx.plot_approx(x, y, y_approx, 'sin(2πx)', lo, hi)