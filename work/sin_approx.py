import numpy as np
import approx_helpers as approx

scale = lambda x: (8 * x / (3 * np.pi) - 8 * x**3 / (3 * np.pi**3))
f = lambda x: np.where(np.abs(x) < 1e-10, 3 * np.pi / 8, np.sin(x) / scale(x))
order = 10
lo = -np.pi+1e-7
hi = np.pi-1e-7

m_coeffs = approx.calc_minimax(f, order, lo, hi, min_rel=False)

lo = -np.pi
hi = np.pi
x = np.linspace(lo, hi, 1000)
y = np.sin(x)
y_approx = scale(x) * approx.eval_poly(x, m_coeffs)

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