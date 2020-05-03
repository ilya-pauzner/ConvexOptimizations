import oracle
import numpy as np


def psi_cup(x0, L, addend, oracle_):
    quotient = 1 / (2 * np.linalg.norm(oracle_.func(x0)))
    loss = (L / 2) * np.linalg.norm(addend) ** 2
    loss += quotient * np.linalg.norm(oracle_.func(x0)) ** 2
    loss += quotient * np.linalg.norm(oracle_.func(x0) + oracle_.grad(x0) * addend) ** 2
    return loss


def optimize_psi_cup(x0, i, oracle_, magic_const):
    best_y = None
    best_y_loss = None
    L = 1
    for j in range(magic_const):
        addend = np.random.rand(*x0.shape)
        addend /= np.linalg.norm(addend)
        addend *= np.linalg.norm(x0)
        addend /= (i ** 1.5 + 15)

        loss = psi_cup(x0, L, addend, oracle_)

        if best_y_loss is None or best_y_loss > loss:
            best_y = x0 + addend
            best_y_loss = loss
    return best_y, best_y_loss


# here func is F, not f=tf.norm(F)
def do_3squares(func, dimension, x0=None, max_iter=1000, tolerance=2e-3, eta=5e-1, magic_const=100):
    losses = []
    if x0 is None:
        x0 = np.array([0.57179] * dimension, dtype=np.float64)
    oracle_ = oracle.BaseSmoothOracle(func)

    for i in range(max_iter):
        best_y, best_y_loss = optimize_psi_cup(x0, i, oracle_, magic_const)

        if np.linalg.norm(best_y - x0) < tolerance:
            break
        x0 = best_y

        loss = np.linalg.norm(oracle_.func(x0))
        print("point:", x0, "loss:", loss, "iter:", i)
        losses.append(loss)

    return x0


if __name__ == '__main__': # testing
    print(do_3squares(lambda x: oracle.apply_func_F(x, oracle.trigonometry), 2, eta=1e-2))
    print(do_3squares(lambda x: oracle.apply_func_F(x, oracle.chebyshev), 2, eta=1e-2))

