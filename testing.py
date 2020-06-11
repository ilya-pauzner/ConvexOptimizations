from method_final import *
import tensorflow as tf
import math


def run_test(n, p, func="lambda x: -tf.math.cos(x * math.pi)"):
    funcs = [lambda x: abs(x[0] - 1)]
    funcs += [eval('lambda x: abs(x[%d] - (%s)(x[%d]))' % (i, func, i - 1)) for i in range(1, n)]
    f_1_cup = lambda x: tf.norm([func(x) for func in funcs])
    x_k, iters, losses = do_method(funcs, n, BaseSmoothOracle(f_1_cup), p=p)
    return iters, losses[-1]


if __name__ == '__main__':
    run_test(10, 5)