from random import randint, shuffle

import tensorflow as tf

from oracle import *


def do_method(funcs, dimension, f_1_cup, p=None, oracles=None, x0=None, max_iter=1000, tolerance=2e-3, do_print=True):
    losses = []
    if x0 is None:
        x0 = np.array([0.57179] * dimension, dtype=np.float64)
    if oracles is None:
        oracles = [BaseSmoothOracle(func) for func in funcs]
    L = 1
    if p is None:
        p = randint(1, len(funcs) - 1)
    batch = [i for i in range(len(funcs))]
    iter = 0
    y = None
    prev_x = None
    while iter < max_iter:
        if y is not None:
            x0 = y
        iter += 1
        shuffle(batch)
        curr = batch[:p]
        G = np.vstack([list(oracles[i].grad(x0)) for i in curr]).T
        i_k = 0
        while True:
            gamma = (2 ** i_k) * L * f_1_cup(x0)
            gamma = 1 / gamma
            B_inv = np.linalg.inv(np.eye(G.shape[0]) + gamma * G.dot(G.T)) / L / (2 ** i_k)
            T = x0 - B_inv.dot(tuple(np.array(f_1_cup.grad(x0)).T)).reshape(x0.shape)
            losses.append(f_1_cup(T))
            if do_print:
                print('iter:', iter, 'loss:', losses[-1])
            phi = lambda y: f_1_cup(x0) + np.dot(f_1_cup.grad(x0), y - x0) + \
                            (np.linalg.norm(G.T.dot(y - x0)) ** 2) / p / 2 / f_1_cup(x0)
            psi = lambda y: phi(y) + 2 ** i_k * L / 2 * np.linalg.norm(y - x0) ** 2
            if f_1_cup(T) <= psi(T):
                break
            i_k += 1
        if np.linalg.norm(x0 - T) < tolerance:
            break

        prev_x = x0
        x0 = T
        if y is None:
            y = T
        else:
            @functools.lru_cache()
            def psi_k(theta):
                x1 = x0 + theta * (x0 - prev_x)
                return f_1_cup(x1)

            @functools.lru_cache()
            def psi_k_derivative(theta):
                return (psi_k(theta + 0.01) - psi_k(theta - 0.01)) / 0.02

            alpha = 1 / 3
            beta = 2 / 3

            theta_best = None

            theta1 = 1
            theta2 = 1
            theta = 1
            if psi_k(0) + beta * psi_k_derivative(0) * theta <= psi_k(theta) <= psi_k(0) + alpha * psi_k_derivative(
                    0) * theta:
                theta_best = theta
            else:
                # localisation
                iters = 0
                while True:
                    iters += 1
                    if iters > 10:
                        break
                    first = psi_k(0) + beta * psi_k_derivative(0) * theta1 > psi_k(theta1)
                    second = psi_k(0) + alpha * psi_k_derivative(0) * theta2 < psi_k(theta2)
                    if first:
                        if second:
                            break
                        else:
                            theta1 = theta2
                            theta2 = 2 * theta1
                    else:
                        maybe_theta1 = theta2 / 2
                        maybe_theta2 = 1

                        maybe_first = psi_k(0) + beta * psi_k_derivative(0) * maybe_theta1 > psi_k(maybe_theta1)
                        maybe_second = psi_k(0) + alpha * psi_k_derivative(0) * maybe_theta2 < psi_k(maybe_theta2)
                        if maybe_first and maybe_second:
                            break
                        else:
                            theta1 = theta2 / 2
                            theta2 = theta1

                # specification
                iters = 0
                while True:
                    iters += 1
                    if iters > 10:
                        theta_best = theta
                        break
                    theta = (theta1 + theta2) / 2
                    if psi_k(0) + beta * psi_k_derivative(0) * theta <= psi_k(theta) <= psi_k(
                            0) + alpha * psi_k_derivative(0) * theta:
                        theta_best = theta
                        break
                    else:
                        if psi_k(0) + beta * psi_k_derivative(0) * theta > psi_k(theta):
                            theta1 = theta
                        else:
                            theta2 = theta

            y = x0 + theta_best * (x0 - prev_x)

        L *= 2 ** (i_k - 1)
    return x0, iter, losses


if __name__ == "__main__":
    f1 = lambda x: 2.7 ** x[0] + 2.7 ** (-x[0])
    f2 = lambda x: abs(x[1])
    f_norm = lambda x: tf.norm([f1(x), f2(x)])
    print(do_method([f1, f2], 2, BaseSmoothOracle(f_norm), p=1)[0])
