import numpy as np
import matplotlib.pyplot as plt
import approx_helpers as approx
from sympy import symbols, asin

# f = lambda x : np.arcsin(np.sqrt(x)) / np.sqrt(x)
# order = 9
# approx_lo = 0
# approx_hi = 1

# m_coeffs = approx.calc_minimax(f, order, approx_lo, approx_hi, a=0, zero_tol=1e-6)

x = symbols('x')
t = approx.calc_taylor(asin(x), x, 0, 13)
p, q = approx.calc_pade(t.coeffs, denom_deg=6)

lo = -1
hi = 1
x = np.linspace(lo, hi, 1000)
y = np.arcsin(x)
# y_approx = approx.eval_poly(x**2, m_coeffs) * x
y_approx = p(x) / q(x)

# Plot the results (three subplots)
plt.subplot(1, 3, 1)
plt.plot(x, y, label='asin(x)')
plt.plot(x, y_approx, label='approx')
plt.legend()
plt.title('Approximation of asin(x) on [-1, 1]')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

# Plot relative error
plt.subplot(1, 3, 2)
rel_error = (y - y_approx) / y
plt.plot(x, rel_error, label='Relative Error', color='red')
plt.legend()
plt.title('Relative Approximation Error')
plt.xlabel('x')
plt.ylabel('Rel. Error')
plt.grid()

# Plot absolute error
plt.subplot(1, 3, 3)
abs_error = np.abs(y - y_approx)
plt.plot(x, abs_error, label='Absolute Error', color='green')
plt.legend()
plt.title('Absolute Approximation Error')
plt.xlabel('x')
plt.ylabel('Abs. Error')
plt.grid()

# set figure size
plt.gcf().set_size_inches(10, 4)
plt.show()

#print(f"Arcsine approximation, order {order} on [{approx_lo}, {approx_hi}]")

#print("Coefficients:")
#print(m_coeffs)

print("Numerator coefficients:")
print(p.coeffs)
print("Denominator coefficients:")
print(q.coeffs)

max_abs_error = np.max(np.abs(y - y_approx))
max_rel_error = np.max(np.abs(y - y_approx) / np.abs(y))
print(f"Max absolute error: {max_abs_error}")
print(f"Max relative error: {max_rel_error}")