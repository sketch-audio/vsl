import numpy as np
from sympy import *
from scipy.interpolate import pade
import cvxpy as cp

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
def calc_minimax(fn, deg, lo, hi, a=0, m=10000, zero_tol=1e-6):
    # create arrays of x and y values
    x = np.linspace(lo, hi, m)
    y = fn(x)

    # remove y values close to zero
    x = x[np.abs(y - 0) > zero_tol]
    y = y[np.abs(y - 0) > zero_tol]

    # if a == 0, we need a constant term
    n = deg + 1 if a == 0 else deg
    stop = -1 if a == 0 else 0
    def X(x): return [(x - a)**i for i in range(deg, stop, -1)]

    # create a vector of coefficients for our polynomial
    A = cp.Variable(n)

    # get the x-part of the polynomial
    V = X(x)

    # create the objective
    obj = cp.max(cp.abs(cp.multiply(A @ cp.transpose(V) - y, 1 / y)))

    # create some optional constraints
    cons = [] # [A @ X(0) == fn(0), etc.]

    # define the problem and solve
    problem = cp.Problem(cp.Minimize(obj), cons)
    problem.solve(solver=cp.CLARABEL)

    # return coefficients
    return A.value