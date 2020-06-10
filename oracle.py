import numpy as np
from func_grad_hess import *


class BaseSmoothOracle(object):
    def __init__(self, function):
        self.function = function

    def func(self, x):
        """
        Computes the value of function at point x.
        """
        return count(tuple(x), self.function)[0]

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        return count(tuple(x), self.function)[1]

    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        return count(tuple(x), self.function)[2]

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


def apply_func_F(x, phi):
    # [1, 2, 3, 4] -> [4, 1, 2, 3]
    y = tf.roll(x, shift=1, axis=0)

    # applying func
    z = phi(y)

    # assigning one to z[0]
    mask_python = [False] * x.shape[0]
    mask_python[0] = True
    mask = tf.constant(mask_python, dtype=tf.bool)
    ones = tf.ones(x.shape[0])
    fixed_z = tf.where(mask, ones, z)

    # returning norm of x - fixed_z
    return x - fixed_z


def apply_func(x, phi, mask_python=None):
    F = apply_func_F(x, phi)
    if mask_python is None:
        mask_python = [True] * x.shape[0]
    mask = tf.constant(mask_python, dtype=tf.bool)
    masked_F = tf.boolean_mask(F, mask)
    return tf.norm(masked_F)


def chebyshev(x):
    return 2 * x * x - 1


def trigonometry(x):
    return -tf.math.cos(np.pi * x)


if __name__ == '__main__':
    x = [25.3, 20.2, 30.1, 27.6]

    oracle_cheb = BaseSmoothOracle(lambda x: apply_func(x, chebyshev))
    print("Chebyshev")
    print(oracle_cheb.func(x))
    print(oracle_cheb.grad(x))
    print(oracle_cheb.hess(x))
    print("Chebyshev")

    print()

    oracle_trig = BaseSmoothOracle(lambda x: apply_func(x, trigonometry))
    print("Trigonometry")
    print(oracle_trig.func(x))
    print(oracle_trig.grad(x))
    print(oracle_trig.hess(x))
    print("Trigonometry")
