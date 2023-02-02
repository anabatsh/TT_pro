import sklearn


import jax.numpy as jnp
import numpy as np
import sys
import teneva
from time import perf_counter as tpc


from protes import protes
from baselines import bs_control_gekko
from baselines import bs_nevergrad
from baselines import bs_optima_tt
from baselines import bs_ttopt
from build_control import build_control
from build_knapsack import build_knapsack
from build_qubo import build_qubo
from utils import Log
from utils import folder_ensure


def calc_control(d_list=[25, 50, 100], M=int(1.E+4), constr=False):
    """Perform computations for optimal control problem."""
    nm = 'CONTROL' + ('_CONSTR' if constr else '')
    log = Log(f'result/logs_check/control{"_constr" if constr else ""}.txt')
    log(f'--> {nm} | d={d_list} | M={M:-7.1e}')

    for d in d_list:
        f, args = build_control(d, constr=constr)
        f_batch = lambda I: np.array([f(i) for i in I])

        t = tpc()
        n_opt, y_opt = protes(f, d, 2, M, constr=constr,
            log=True, log_ind=True)
        t = tpc() - t

        text = ''
        text += f' ODE-{d}D | '
        text += f'JAX v={y_opt:-9.2e}, t={t:-9.2e} | '
        log(text)


def calc_func(d=7, n=16, M=1.E+4, M_ng=1.E+4, with_shift=True):
    """Perform computations for analytical multivariable functions."""
    M = int(M)
    M_ng = int(M_ng or M)

    nm = 'FUNC'
    log = Log(f'result/logs_check/func.txt')
    log(f'--> {nm} | d={d} | n={n} | M={M:-7.1e} | Mng={M_ng:-7.1e}')

    names = ['Ackley', 'Alpine', 'Exponential', 'Grienwank', 'Michalewicz',
        'Piston', 'Qing', 'Rastrigin', 'Schaffer', 'Schwefel']

    for func in teneva.func_demo_all(d, names=names, with_piston=True):
        # Set the grid:
        func.set_grid(n, kind='uni')

        # Translate the function limits to ensure correct competition:
        if with_shift:
            shift = np.random.randn(d) / 10
            a_new = func.a - (func.b-func.a) * shift
            b_new = func.b + (func.b-func.a) * shift
            func.set_lim(a_new, b_new)

        # Target function for optimization:
        f = func.get_f_ind

        t = tpc()
        n_opt, y_opt = protes(f, d, n, M, batch=True,
            log=True)
        t = tpc() - t

        # Present the result:
        text = ''
        text += func.name + ' ' * max(0, 12-len(func.name)) +  ' | '
        text += f'JAX v={y_opt:-9.2e}, t={t:-9.2e} | '
        log(text)


def calc_knapsack(M=1.E+4, M_ng=1.E+4):
    """Perform computations for concrete Knapsack problem."""
    M = int(M)
    M_ng = int(M_ng or M)

    nm = 'KNAPSACK'
    log = Log(f'result/logs_check/knapsack.txt')
    log(f'--> {nm} M={M:-7.1e} | Mng={M_ng:-7.1e}')

    for task in [1]:
        d, f = build_knapsack()
        f_batch = lambda I: np.array([f(i) for i in I])

        t = tpc()
        n_opt, y_opt = protes(f_batch, d, 2, M,
            batch=True, log=True)
        t = tpc() - t

        # Present the result:
        text = ''
        text += f' kn-{task} | '
        text += f'JAX v={y_opt:-9.2e}, t={t:-9.2e} | '
        log(text)


def calc_qubo(d=50, M=1.E+4, M_ng=1.E+4):
    """Perform computations for QUBO problem."""
    M = int(M)
    M_ng = int(M_ng or M)

    nm = 'QUBO'
    log = Log(f'result/logs_check/qubo.txt')
    log(f'--> {nm} | d={d} | M={M:-7.1e} | Mng={M_ng:-7.1e}')

    for task in [1, 2, 3]:
        f_batch = build_qubo(d, task)
        f = lambda i: f_batch(np.array(i).reshape(1, -1))[0]

        t = tpc()
        n_opt, y_opt = protes(f_batch, d, 2, M,
            batch=True, log=True)
        t = tpc() - t

        # Present the result:
        text = ''
        text += f' QUBO-{task} | '
        text += f'JAX v={y_opt:-9.2e}, t={t:-9.2e} | '
        log(text)


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs_check')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'func'

    if mode == 'control':
        calc_control()
    elif mode == 'control_constr':
        calc_control(constr=True)
    elif mode == 'func':
        calc_func()
    elif mode == 'knapsack':
        calc_knapsack()
    elif mode == 'qubo':
        calc_qubo()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
