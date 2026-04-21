import numpy as np
import approx_helpers as approx

scale = lambda x: (-4 * x**2 / (3 * np.pi) + np.pi / 3)
f = lambda x: np.cos(x) / scale(x)
order = 10
lo = -np.pi
hi = np.pi

m_coeffs = approx.calc_minimax(f, order, lo, hi, min_rel=False)

x = np.linspace(lo, hi, 1000)
y = np.cos(x)
y_approx = scale(x) * approx.eval_poly(x, m_coeffs)

print(f"Cosine approximation, order {order} on [{lo}, {hi}]")
approx.print_coeffs(m_coeffs, "Minimax")
approx.print_max_abs_error(x, y, y_approx)
approx.print_max_rel_error(x, y, y_approx)

approx.plot_approx(x, y, y_approx, 'cos(x)', lo, hi)

# min pin zero
m_abs = approx.calc_minimax(f, order, lo, hi, pin_zero=True)
y_abs = scale(x) * approx.eval_poly(x, m_abs)
print(f"\nCosine approximation (min pin zero), order {order} on [{lo}, {hi}]")
approx.print_coeffs(m_abs, "Min Pin Zero")
approx.print_max_abs_error(x, y, y_abs)
approx.print_max_rel_error(x, y, y_abs)

approx.plot_approx(x, y, y_abs, 'cos(x) min pin zero', lo, hi)

# cos(2πx)
f = lambda x: np.cos(2 * np.pi * x)
order = 8

lo = -0.5
hi = 0.5

m_coeffs = approx.calc_minimax(f, order, lo, hi)

x = np.linspace(lo, hi, 1000)
y = f(x)
y_approx = approx.eval_poly(x, m_coeffs)

print(f"\nCosine(2πx) approximation, order {order} on [{lo}, {hi}]")
approx.print_coeffs(m_coeffs, "Minimax")
approx.print_max_abs_error(x, y, y_approx)
approx.print_max_rel_error(x, y, y_approx)

approx.plot_approx(x, y, y_approx, 'cos(2πx)', lo, hi)
