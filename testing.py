import math
from time import time
import tensorflow as tf

import method_final
import method_final_momentum
from oracle import *


test_func_1 = "lambda x: -tf.math.cos(x * math.pi)"
test_func_2 = "lambda x: 2 * x * x - 1"


class Oracle1(BaseSmoothOracle):
    def __init__(self, i):
        super().__init__(lambda x: x / 0)
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


class Oracle2(BaseSmoothOracle):
    def __init__(self, i):
        super().__init__(lambda x: x / 0)  # underlying class methods should never be called
        self.i = i

    def func(self, x):
        if self.i == 0:
            return x[0] - 1
        else:
            return x[self.i] - 2 * (x[self.i - 1] ** 2) + 1

    def grad(self, x):
        if self.i == 0:
            return [1] + [0] * (len(x) - 1)
        else:
            res = [0] * len(x)
            res[self.i] = 1
            res[self.i - 1] = - 4 * x[self.i - 1]
            return res


def run_test(n, p, func=test_func_1):
    funcs = [lambda x: (x[0] - 1)]
    funcs += [eval('lambda x: (x[%d] - (%s)(x[%d]))' % (i, func, i - 1)) for i in range(1, n)]
    f_1_cup = lambda x: tf.norm([func(x) for func in funcs])
    oracles = None
    if func == test_func_1:
        oracles = [Oracle1(i) for i in range(n)]
    if func == test_func_2:
        oracles = [Oracle2(i) for i in range(n)]
    f_1_cup = BaseSmoothOracle(f_1_cup)
    start_time = time()
    x_k, iters, losses = method_final.do_method(funcs, n, f_1_cup, p=p, oracles=oracles, do_print=False)
    print('test of method without moments with n = %d and p = %d' % (n, p), 'and func %s' % func, 'gave the folowing results:')
    print('elapsed:', time() - start_time)
    print('min loss:', min(losses))
    print('iterations:', iters)
    print('calls to oracles:', f_1_cup.func_calls + f_1_cup.grad_calls + sum([oracle.func_calls + oracle.grad_calls for oracle in oracles]))
    print()


def run_test_momentum(n, p, func=test_func_1):
    funcs = [lambda x: (x[0] - 1)]
    funcs += [eval('lambda x: (x[%d] - (%s)(x[%d]))' % (i, func, i - 1)) for i in range(1, n)]
    f_1_cup = lambda x: tf.norm([func(x) for func in funcs])
    oracles = None
    if func == test_func_1:
        oracles = [Oracle1(i) for i in range(n)]
    if func == test_func_2:
        oracles = [Oracle2(i) for i in range(n)]
    x_k, iters, losses = method_final_momentum.do_method(funcs, n, BaseSmoothOracle(f_1_cup), p=p, oracles=oracles)
    return iters, losses[-1]


if __name__ == '__main__':
    print("FINAL METHOD")
    run_test(10, 5, func=test_func_2)
    run_test(10, 5)
    print("FINAL METHOD MOMENTUM")
    run_test_momentum(10, 5, func=test_func_2)
    run_test_momentum(10, 5)