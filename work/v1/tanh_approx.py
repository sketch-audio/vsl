import numpy as np
import approx_helpers as approx
from sympy import symbols, tanh

x = symbols('x')
t = approx.calc_taylor(tanh(x), x, 0, 13)
p, q = approx.calc_pade(t.coeffs, denom_deg=6)

k = 5
lo = -k
hi = k
x = np.linspace(lo, hi, 1000)
y = np.tanh(x)
y_approx = p(x) / q(x)

print(f"Tanh approximation on [{lo}, {hi}]")
approx.print_coeffs(p.coeffs, "Numerator")
approx.print_coeffs(q.coeffs, "Denominator")
approx.print_max_abs_error(x, y, y_approx)
approx.print_max_rel_error(x, y, y_approx)

approx.plot_approx(x, y, y_approx, 'tanh(x)', lo, hi)