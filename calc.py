import numpy as np
import sys
import teneva
from time import perf_counter as tpc


from tt_pro import tt_pro
from utils import Log
from utils import folder_ensure


def calc_test(d=10, n=100, M=1.E+4, K=20, k=1, k_gd=100, r=5, M_ANOVA=None):
    """Perform simple computations to test / check the TT-PRO method."""
    log = Log(f'result/logs/test.txt')
    time = tpc()

    txt = f'--> TEST | '
    txt += f'd={d} n={n} M={M:-7.1e} K={K} k={k} k_gd={k_gd} r={r}'
    log(txt)

    # Target function ('Ackley', 'Alpine', 'Dixon', 'Exponential',
    # 'Grienwank', 'Michalewicz', 'Qing', 'Rastrigin', 'Schaffer', 'Schwefel'):
    func = teneva.func_demo_all(d, names=['Schaffer'])[0]
    func.set_grid(n, kind='uni')
    f = func.get_f_ind

    n_opt = tt_pro(f, d, n, M, K, k, k_gd, r, M_ANOVA, batch=True, log=True)
    y_opt = f(n_opt)

    log(f'\n--------')
    log(f'Result : {y_opt:-14.7e}')
    log(f'Time   : {tpc()-time:-14.3f}')


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'

    if mode == 'test':
        calc_test() # M_ANOVA=1.E+3
    elif mode == 'qubo':
        pass
        # TODO: здесь будут разные варианты запуска (разные задачи)
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
