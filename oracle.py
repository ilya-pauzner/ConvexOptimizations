import numpy as np
from func_grad_hess import *


class BaseSmoothOracle(object):
    def __init__(self, function):
        self.function = function

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        return count(x, self.function)[0]

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        return count(x, self.function)[1]

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        return count(x, self.function)[2]

    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)
