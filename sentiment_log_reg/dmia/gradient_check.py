from typing import Callable

import numpy as np
from random import randrange


def eval_numerical_gradient(f: Callable, x: np.ndarray):
    """A naive implementation of numerical gradient of f at x

    Args:
        f: should be a function that takes a single argument
        x: is the point (numpy array) to evaluate the gradient at
    """

    # Evaluates function value at original point
    fx = f(x)
    grad = np.zeros(x.shape)
    h = 0.00001

    # Iterates over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        # Increments by h
        x[ix] += h
        # Evaluates f(x + h)
        fxh = f(x)
        # Restores to previous value (very important!)
        x[ix] -= h

        # Computes the partial derivative
        # The slope
        grad[ix] = (fxh - fx) / h
        print(ix, grad[ix])
        # Steps to the next dimension
        it.iternext()
    return grad


def grad_check_sparse(f, x, analytic_grad, num_checks):
    """Samples a few random elements and only return numerical in dimensions"""
    h = 1e-5

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])
        # Increments by h
        x[ix] += h
        # Evaluates f(x + h)
        fxph = f(x)
        # Increments by h
        x[ix] -= 2 * h
        # Evaluates f(x - h)
        fxmh = f(x)
        # Resets
        x[ix] += h

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = (
            abs(grad_numerical - grad_analytic)
            / (abs(grad_numerical) + abs(grad_analytic))
        )
        print(
            'numerical: %f analytic: %f, relative error: %e' % (
                grad_numerical, grad_analytic, rel_error
            )
        )
