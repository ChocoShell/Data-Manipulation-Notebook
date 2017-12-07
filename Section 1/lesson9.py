# Optimizers

# What is an Optimizer?
"""
Can be used to find minimum values of functions

Build Parametrized models based on data

Refine allocations to stocks in portfolios
"""

# How to use an optimizer
"""
1) Prove a function to minimize

2) Provide an initial guess

3) Call the optimizer
"""

# Minimization example
"""
f(x) = (x-1.5)^2 + 0.5

Lets let the minimizer start at 2.0
It will check the value at 2.0 then the value at less and more than the initial.
It gets a slope and tries to continue doing this with gradient descent.
"""

# Minimizer in python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

# def f(x):
#     """Given a scalar, x, return some value (a real number)."""
#     y = (x-1.5)**2 + 0.5
#     print("X = {}, Y = {}".format(x, y))
#     return y

# def test_run():
#     xguess = 2.0
#     min_result = spo.minimize(f, xguess, method='SLSQP', options={'disp': True})
#     print("X = {}, Y = {}".format(min_result.x, min_result.fun))

#     # Plotting
#     xplot = np.linspace(0.5, 2.5, 21)
#     yplot = f(xplot)
#     plt.plot(xplot, yplot)
#     plt.plot(min_result.x, min_result.fun, 'ro')
#     plt.title("Minima of an objective function")
#     plt.show()

# How to defeat a minimizer
"""
Which functions would be hard to solve?
Functions with multiple minima,
    any discontinuities or zero slope 
    will be hard for our minimizer.
"""

# Convex Problems
"""
Most easy for optimizers to solve.
Covnex Problem - 
    A real value function defined on an interval,
    if any 2 points if a line is made connecting them
    is above the graph, it is convex.  The graph cannot 
    cross any line.

Optimizers can work in multiple dimensions (3D)
"""

# Building a parametrized model
"""
Parameterized model - f(x) = mx + b
    has two parameters m and b (C0 and C1)

What are we trying to minimize finding the best parameters?

We can see how far the dots are from our lines and mark them as error.
"""

# Minimizer finds coefficients.
"""

"""
def error(line, data):
    """ Compute error between given line model and observed data.

    Parameters
    ----------
    line: tuple/list/array C0, C1 where C0 is the 
        slope and C1 is the y intercept
    data: 2D array where each row is a point (x, y)

    Returns error as a single real value.
    """
    # Metric: SUm of squared Y-axis differences
    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)
    return err

# And it works for polynomials too!
def error_poly(C, data):
    err = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)
    return err

def fit_line(data, error_func):
    """Fit a line to given data, using a supplied error funciton.

    Parameters
    ----------
    data: 2D array where each row is a point (X0, Y)
    error_func: function that computes the error between
        a line and observed data

    Returns line that minimizes error function.
    """
    # Generate initial guess for line model
    l = np.float32([0, np.mean(data[:, 1])]) # slope = 0, interecept = mean(y values)

    # Plot initial guess (optional)
    x_ends = np.float32([-5, 5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--',
        linewidth=2.0, label="Initial Guess")

    # Call optimizer to minimize error function
    result = spo.minimize(error_func, l, args=(data,),
        method='SLSQP',
        options={'disp': True})
    return result.x

def fit_poly(data, error_func, degree=3):
    Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

    x = np.linespace(-5, 5, 21)
    plt.plot(x, np.polyval(cguess, x), 'm--', label="Initial Guess")

    result = spo.minimize(error_func, Cguess, args=(data,),
        method='SLSQP',
        options={'disp': True})
    return np.poly1d(result.x)

def test_run():
    # Define original line
    l_orig = np.float32([4, 2])
    print("Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1]))
    xorig = np.linspace(0, 10, 21)
    yorig = l_orig[0] * xorig + l_orig[1]
    plt.plot(xorig, yorig, 'b--', linewidth=2.0, label="Original line")

    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, yorig.shape)
    data = np.asarray([xorig, yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label="Data points")

    # Try to fit a line to this data
    l_fit = fit_line(data, error)
    print("Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1]))
    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--',
        linewidth=2.0, label="New Line")

    plt.show()

# Review
"""
We minimized in multiple dimensions and in parametrized models.
"""

if __name__ == "__main__":
    test_run()
