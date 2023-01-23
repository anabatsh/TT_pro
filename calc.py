import numpy as np
import sys
import teneva
from time import perf_counter as tpc


from qubo import qubo_build_function
from qubo import qubo_build_matrix
from qubo import qubo_solve_baseline
from tt_pro import tt_pro
from utils import Log
from utils import folder_ensure


def calc_control():
    raise NotImplementedError('TODO')


def calc_qubo(d=100, M=1.E+4, K=20, k=4, k_gd=100, r=5):
    log = Log(f'result/logs/qubo.txt')

    txt = f'--> QUBO | '
    txt += f'd={d} M={M:-7.1e} K={K} k={k} k_gd={k_gd} r={r}'
    log(txt)

    Q = qubo_build_matrix(d, prob_con=0.5)

    t_ref = tpc()
    y_opt_ref = qubo_solve_baseline(Q)
    t_ref = tpc() - t_ref

    f = qubo_build_function(np.array(Q))
    info = {}
    n_opt = tt_pro(f, d, 2, M, K, k, k_gd, r, info=info, batch=True, log=True)
    y_opt = f(n_opt.reshape(1, -1))[0]

    log(f'\n--------')
    log(f'Result : {y_opt:-14.7e} | Baseline: {y_opt_ref:-14.7e}')
    log(f'Time   : {info["t"]:-14.3f} | Baseline: {t_ref:-14.7e}')


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

    info = {}
    n_opt = tt_pro(f, d, n, M, K, k, k_gd, r, M_ANOVA, info=info,
        batch=True, log=True)
    y_opt = f(n_opt)

    log(f'\n--------')
    log(f'Result : {y_opt:-14.7e}')
    log(f'Time   : {info["t"]:-14.3f}')


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'

    if mode == 'control':
        calc_control()
    elif mode == 'qubo':
        calc_qubo()
    elif mode == 'test':
        calc_test() # M_ANOVA=1.E+3
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
