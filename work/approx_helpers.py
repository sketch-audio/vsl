import numpy as np
from sympy import *
from scipy.interpolate import pade
import cvxpy as cp
import matplotlib.pyplot as plt

# Evaluate a polynomial.
def eval_poly(x, coeffs, a=0):
    start = len(coeffs) - 1 if a == 0 else len(coeffs)
    stop = -1 if a == 0 else 0
    v = np.array([(x - a)**i for i in range(start, stop, -1)])  # a column for each x-part
    y = [v[:,i] * coeffs for i in range(len(x))]                # multiply columns by coeffs
    return np.sum(y, axis=1)                                    # sum the columns

# Return a numpy polynomial
def calc_taylor(expr, var, ctr, deg):
    ser = expr.series(var, ctr, deg + 1).removeO().expand()
    pol = Poly(ser, var)
    cfs = [float(c.evalf()) for c in pol.all_coeffs()]
    return np.poly1d(cfs)

# Return two numpy polynomials (numerator, denominator)
def calc_pade(coeffs_desc, denom_deg):
    p, q = pade(coeffs_desc[::-1], denom_deg)
    return p, q

# Minimizes maximum relative error.
def calc_minimax(fn, deg, lo, hi, a=0, m=10000, zero_tol=1e-6, min_rel=True):
    # create arrays of x and y values
    x = np.linspace(lo, hi, m)
    y = fn(x)

    # remove y values close to zero
    if not min_rel:
        zero_tol = 0
    
    mask = np.abs(y) > zero_tol
    x = x[mask]
    y = y[mask]

    # if a == 0, we need a constant term
    n = deg + 1 if a == 0 else deg
    stop = -1 if a == 0 else 0
    def X(x): return [(x - a)**i for i in range(deg, stop, -1)]

    # create a vector of coefficients for our polynomial
    A = cp.Variable(n)

    # get the x-part of the polynomial
    V = X(x)

    # create the objective
    if not min_rel:
        obj = cp.max(cp.abs(A @ cp.transpose(V) - y))
    else:
        obj = cp.max(cp.abs(cp.multiply(A @ cp.transpose(V) - y, 1 / y)))

    # create some optional constraints
    cons = [] # [A @ X(0) == fn(0), etc.]

    # define the problem and solve
    problem = cp.Problem(cp.Minimize(obj), cons)
    problem.solve(solver=cp.CLARABEL)

    # return coefficients
    return A.value


def plot_approx(x, y, y_approx, f_name, x_min, x_max):
    # Plot the results (three subplots)
    plt.subplot(1, 3, 1)
    plt.plot(x, y, label=f_name)
    plt.plot(x, y_approx, label='approx')
    plt.legend()
    plt.title(f'Approximation of {f_name} on [{x_min}, {x_max}]')
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

def print_coeffs(coeffs, name):
    print(f"{name} coefficients:")
    print(coeffs)

# Print max absolute error and input which produces it.
def print_max_abs_error(x, y, y_approx):
    abs_error = np.abs(y - y_approx)
    max_abs_error = np.max(abs_error)
    max_index = np.argmax(abs_error)
    print(f"Max absolute error: {max_abs_error:.6f} at x = {x[max_index]:.6f}")
    print(f"Expected y: {y[max_index]:.6f}, Approx y: {y_approx[max_index]:.6f}")

# Print max relative error and input which produces it.
def print_max_rel_error(x, y, y_approx):
    rel_error = np.abs(y - y_approx) / np.abs(y)
    max_rel_error = np.max(rel_error)
    max_index = np.argmax(rel_error)
    # Print error as percent, show expected y value and approximated y value
    print(f"Max relative error: {max_rel_error * 100:.2f}% at x = {x[max_index]:.6f}")
    print(f"Expected y: {y[max_index]:.6f}, Approx y: {y_approx[max_index]:.6f}")