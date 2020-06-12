from method_final import *
import tensorflow as tf
import math

class Oracle1(BaseSmoothOracle):
    def __init__(self, i):
        self.i = i

    def func(self, x):
        if self.i == 0:
            return x[0] - 1
        else:
            return x[self.i] + math.cos(math.pi * x[self.i - 1])

    def grad(self, x):
        if self.i == 0:
            return [1] + [0] * (len(x) - 1)
        else:
            res = [0] * len(x)
            res[self.i] = 1
            res[self.i - 1] = - math.pi * math.sin(math.pi * x[self.i - 1])
            return res


def run_test(n, p, func="lambda x: -tf.math.cos(x * math.pi)"):
    funcs = [lambda x: abs(x[0] - 1)]
    funcs += [eval('lambda x: (x[%d] - (%s)(x[%d]))' % (i, func, i - 1)) for i in range(1, n)]
    f_1_cup = lambda x: tf.norm([func(x) for func in funcs])
    oracles = None
    if func == "lambda x: -tf.math.cos(x * math.pi)":
        oracles = [Oracle1(i) for i in range(n)]
    x_k, iters, losses = do_method(funcs, n, BaseSmoothOracle(f_1_cup), p=p, oracles=oracles)
    return iters, losses[-1]


if __name__ == '__main__':
    run_test(10, 5)