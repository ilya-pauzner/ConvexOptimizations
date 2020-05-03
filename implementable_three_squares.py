import numpy as np
import oracle
from three_squares import psi_cup, optimize_psi_cup


# 2.22
def do_implementable_three_squares(func, dimension, x0=None, max_iter=1000, tolerance=2e-3, eta=5e-1, magic_const=100):
    losses = []
    if x0 is None:
        x0 = np.array([0.57179] * dimension, dtype=np.float64)
    oracle_ = oracle.BaseSmoothOracle(func)

    L_0 = 0.5 # or what should it be?
    for k in range(max_iter):
        i_k = 0
        while True:
            best_y, best_loss = optimize_psi_cup(x0, L_0 * (2 ** i_k), k, oracle_, magic_const)
            if np.linalg.norm(oracle_.func(best_y)) <= best_loss:
                break
            i_k += 1
        if np.linalg.norm(best_y - x0) < tolerance:
            break
        x0 = best_y
        loss = np.linalg.norm(oracle_.func(x0))
        print("point:", x0, "loss:", loss, "iter:", k)
        losses.append(loss)
        L_0 *= 2 ** (i_k - 1)
    return x0


if __name__ == '__main__': # testing
    print(do_implementable_three_squares(lambda x: oracle.apply_func_F(x, oracle.trigonometry), 2, eta=1e-2))
    print(do_implementable_three_squares(lambda x: oracle.apply_func_F(x, oracle.chebyshev), 2, eta=1e-2))


