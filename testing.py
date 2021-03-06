import math
import multiprocessing
from time import time

import pandas as pd
import tensorflow as tf

import method_final
import method_final_momentum
import method_final_momentum_armijo
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


def init(l):
    global lock
    lock = l


def print_result(result):
    print('test of method %s with n = %d and p = %d' % (result['method'], result['n'], result['p']),
          'and func %s' % result['func'],
          'gave the folowing results:')
    print('elapsed:', result['elapsed'])
    print('min loss:', result['min_loss'])
    print('iterations:', result['iterations'])
    print('calls to oracles:', result['calls_to_oracles'])
    print()


def run_test(n, p, func=test_func_1, method=method_final.do_method, method_name='without_moments'):
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
    x_k, iters, losses = method(funcs, n, f_1_cup, p=p, oracles=oracles, do_print=False)

    with lock:
        result = {'method': method_name, 'func': '"' + str(func) + '"', 'n': n, 'p': p, 'elapsed': time() - start_time,
                  'min_loss': min(losses), 'iterations': iters,
                  'calls_to_oracles': f_1_cup.func_calls + f_1_cup.grad_calls +
                                      sum([oracle.func_calls + oracle.grad_calls for oracle in oracles])}

        print_result(result)

        return result


def helper(args):
    return run_test(**args)


if __name__ == '__main__':
    args = [
        {'n': 3, 'p': 2, 'func': test_func_1, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 4, 'p': 2, 'func': test_func_1, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 6, 'p': 3, 'func': test_func_1, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 8, 'p': 4, 'func': test_func_1, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 10, 'p': 5, 'func': test_func_1, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 12, 'p': 6, 'func': test_func_1, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 14, 'p': 7, 'func': test_func_1, 'method': method_final.do_method, 'method_name': 'without_moments'},

        {'n': 3, 'p': 2, 'func': test_func_2, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 4, 'p': 2, 'func': test_func_2, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 6, 'p': 3, 'func': test_func_2, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 8, 'p': 4, 'func': test_func_2, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 10, 'p': 5, 'func': test_func_2, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 12, 'p': 6, 'func': test_func_2, 'method': method_final.do_method, 'method_name': 'without_moments'},
        {'n': 14, 'p': 7, 'func': test_func_2, 'method': method_final.do_method, 'method_name': 'without_moments'},

        {'n': 3, 'p': 2, 'func': test_func_1, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 4, 'p': 2, 'func': test_func_1, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 6, 'p': 3, 'func': test_func_1, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 8, 'p': 4, 'func': test_func_1, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 10, 'p': 5, 'func': test_func_1, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 12, 'p': 6, 'func': test_func_1, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 14, 'p': 7, 'func': test_func_1, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},

        {'n': 3, 'p': 2, 'func': test_func_2, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 4, 'p': 2, 'func': test_func_2, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 6, 'p': 3, 'func': test_func_2, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 8, 'p': 4, 'func': test_func_2, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 10, 'p': 5, 'func': test_func_2, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 12, 'p': 6, 'func': test_func_2, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},
        {'n': 14, 'p': 7, 'func': test_func_2, 'method': method_final_momentum.do_method,
         'method_name': 'with_extrapolation_moment'},

        {'n': 3, 'p': 2, 'func': test_func_1, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 4, 'p': 2, 'func': test_func_1, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 6, 'p': 3, 'func': test_func_1, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 8, 'p': 4, 'func': test_func_1, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 10, 'p': 5, 'func': test_func_1, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 12, 'p': 6, 'func': test_func_1, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 14, 'p': 7, 'func': test_func_1, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},

        {'n': 3, 'p': 2, 'func': test_func_2, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 4, 'p': 2, 'func': test_func_2, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 6, 'p': 3, 'func': test_func_2, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 8, 'p': 4, 'func': test_func_2, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 10, 'p': 5, 'func': test_func_2, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 12, 'p': 6, 'func': test_func_2, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
        {'n': 14, 'p': 7, 'func': test_func_2, 'method': method_final_momentum_armijo.do_method,
         'method_name': 'with_armijo_moment'},
    ]

    with multiprocessing.Pool(initializer=init, initargs=(multiprocessing.Lock(),)) as p:
        answer = list(p.map(helper, args))

    df = pd.DataFrame(columns=answer[0].keys())
    df = df.append(answer, ignore_index=True)
    df.to_csv('results.csv')
