from oracle import *


def do_gradient_descent(function, dimension, x0=None, max_iter = 1000, tolerance=1e-4, eta=np.float64(1e-1)):
    iters = 0
    losses = []
    if x0 is None:
        x0 = np.array([1.0] * dimension, dtype=np.float64)
    oracle = BaseSmoothOracle(function)
    while iters < max_iter:
        iters += 1
        losses.append(oracle.func(x0))
        grad = oracle.grad(x0)[0]
        new_x = x0 - grad * eta
        print(new_x)
        if np.linalg.norm(new_x - x0) < tolerance:
            break
        x0 = new_x
    return x0


if __name__ == '__main__': # testing
    print(do_gradient_descent(lambda x: x * x, 1))
    print(do_gradient_descent(lambda x: x[0] * x[0] + x[1] * x[1], 2))
    print(do_gradient_descent(lambda x: 6 * x[0] * x[0] + x[1] * x[1], 2))