import oracle
import numpy as np
import scipy.linalg as sla


def do_newton(func, dimension, x0=None, max_iter=1000, tolerance=1e-4, eta=5e-1):
    losses = []
    if x0 is None:
        x0 = np.array([1.0] * dimension, dtype=np.float64)
    oracle_ = oracle.BaseSmoothOracle(func)

    for i in range(max_iter):
        loss = oracle_.func(x0)
        print(loss)
        losses.append(loss)

        hess = np.array(oracle_.hess(x0)[0])
        hess_inv = sla.inv(hess)
        grad = np.array(oracle_.grad(x0))
        addend = np.dot(hess_inv, grad.T).T
        x1 = (x0 - eta * addend)[0]

        if np.linalg.norm(x1 - x0) < tolerance:
            break

        x0 = x1

    return x0

if __name__ == '__main__': # testing
    print(do_newton(lambda x: x * x, 1))
    print(do_newton(lambda x: x[0] * x[0] + x[1] * x[1], 2))
    print(do_newton(lambda x: 6 * x[0] * x[0] + x[1] * x[1], 2))
