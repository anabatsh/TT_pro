import numpy as np
import sys
import teneva
from time import perf_counter as tpc


from tt_pro import tt_pro
from utils import Log
from utils import folder_ensure


def calc_test(d=10, n=100, M = 2.E+5, K=20, k=1, k_sgd=10, r=5):
    """Perform computations to test the TT-PRO method."""
    log = Log(f'result/logs/calc_test.txt')

    txt = f'---> CALC TEST '
    txt += f'| d: {d:-4d} | n: {n:-5d} | K: {K:-3d} | k: {k:-3d} '
    txt += f'| k_sgd: {k_sgd:-3d} | M: {M:-7.1e} | r: {r:-3d} \n'
    log(txt)

    # Целевая функция (можно взять: 'Ackley', 'Alpine', 'Dixon', 'Exponential',
    # 'Grienwank', 'Michalewicz', 'Qing', 'Rastrigin', 'Schaffer', 'Schwefel'):
    func = teneva.func_demo_all(d, names=['Schaffer'])[0]
    func.set_grid(n, kind='uni')
    f = func.get_f_ind

    # f - это целевая функция, в нашем случае вычисляющая значения
    # дискретизированной функции Schaffer (то есть неявно заданного тензора
    # размерности d с числом узлов n по каждой моде) для заданного набора
    # (батча) мульти-индексов.

    y_opt = tt_pro(f, d, n, M, K, k, k_sgd, r, batch=True, log=True)
    # y_opt = f(n_opt)

    print(y_opt)


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'

    if mode == 'test':
        calc_test()
    # TODO: здесь будут разные варианты запуска (разные задачи)
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
