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
from build_qubo import build_qubo
from utils import Log
from utils import folder_ensure


def calc_control(d_list=[25, 50, 100], M=int(1.E+3), constr=False):
    """Perform computations for optimal control problem."""
    nm = 'CONTROL' + ('_CONSTR' if constr else '')
    log = Log(f'result/logs/control{"_constr" if constr else ""}.txt')
    log(f'--> {nm} | d={d_list} | M={M:-7.1e}')

    for d in d_list:
        f, args = build_control(d, constr=constr)
        f_batch = lambda I: np.array([f(i) for i in I])

        # OWN: Find min value for the original tensor by the proposed method:
        n_opt_own, y_opt_own = protes(f, d, 2, M, constr=constr,
            log=True, log_ind=True)

        # BS1: Find min value the original tensor by TTOpt:
        n_opt_bs1, y_opt_bs1 = bs_ttopt(f_batch, [2]*d, M)
        print(f'BS1 : {y_opt_bs1:-9.2e}')

        # BS2: Find min value for TT-tensor by Optima-TT:
        n_opt_bs2, y_opt_bs2 = bs_optima_tt(f_batch, [2]*d, M)
        print(f'BS2 : {y_opt_bs2:-9.2e}')

        # BS3 OnePlusOne method from nevergrad:
        n_opt_bs3, y_opt_bs3 = bs_nevergrad(f, [2]*d, M, 'OnePlusOne')
        print(f'BS3 : {y_opt_bs3:-9.2e}')

        # BS4 PSO method from nevergrad:
        n_opt_bs4, y_opt_bs4 = bs_nevergrad(f, [2]*d, M, 'PSO')
        print(f'BS4 : {y_opt_bs4:-9.2e}')

        # BS5 PSO method from nevergrad:
        n_opt_bs5, y_opt_bs5 = bs_nevergrad(f, [2]*d, M, 'NoisyBandit')
        print(f'BS5 : {y_opt_bs5:-9.2e}')

        # BS6 PSO method from nevergrad:
        n_opt_bs6, y_opt_bs6 = bs_nevergrad(f, [2]*d, M, 'SPSA')
        print(f'BS6 : {y_opt_bs6:-9.2e}')

        # BS7 PSO method from nevergrad:
        n_opt_bs7, y_opt_bs7 = bs_nevergrad(f, [2]*d, M, 'Portfolio')
        print(f'BS7 : {y_opt_bs7:-9.2e}')

        # BS8: Find min value the original tensor by GEKKO:
        n_opt_bs8, y_opt_bs8 = bs_control_gekko(*args, constr=constr)
        print(f'BS8 : {y_opt_bs8:-9.2e}')

        # Present the result:
        text = ''
        text += f' ODE-{d}D | '
        text += f'OWN {y_opt_own:-9.2e} | ' #
        text += f'BS1 {y_opt_bs1:-9.2e} | ' # TTOpt
        text += f'BS2 {y_opt_bs2:-9.2e} | ' # Optima-TT
        text += f'BS3 {y_opt_bs3:-9.2e} | ' # nevergrad OnePlusOne
        text += f'BS4 {y_opt_bs4:-9.2e} | ' # nevergrad PSO
        text += f'BS5 {y_opt_bs5:-9.2e} | ' # nevergrad NoisyBandit
        text += f'BS6 {y_opt_bs6:-9.2e} | ' # nevergrad SPSA
        text += f'BS7 {y_opt_bs7:-9.2e} | ' # nevergrad Portfolio
        text += f'BS8 {y_opt_bs8:-9.2e} | ' # GEKKO Base method
        log(text)


def calc_func(d=10, n=50, M=2.E+5, M_ng=1.E+3, with_shift=False):
    """Perform computations for analytical multivariable functions."""
    M = int(M)
    M_ng = int(M_ng or M)

    nm = 'FUNC'
    log = Log(f'result/logs/func.txt')
    log(f'--> {nm} | d={d} | n={n} | M={M:-7.1e} | Mng={M_ng:-7.1e}')

    funcs = teneva.func_demo_all(d, only_with_min=True, only_with_min_x=True)
    for func in funcs:
        # Set the grid:
        func.set_grid(n, kind='uni')

        # Translate the function limits to ensure correct competition:
        if with_shift:
            shift = 0.32
            a_new = func.x_min - (func.b-func.a) * shift
            b_new = func.x_min + (func.b-func.a) * (1. - shift)
            func.set_lim(a_new, b_new)

        # Target function for optimization:
        f = func.get_f_ind

        # OWN: Find min value for the original tensor by the proposed method:
        n_opt_own, y_opt_own = protes(f, d, n, M, batch=True,
            log=True)

        # BS1: Find min value the original tensor by TTOpt:
        n_opt_bs1, y_opt_bs1 = bs_ttopt(f, func.n, M)
        print(f'BS1 : {y_opt_bs1:-9.2e}')

        # BS2: Find min value for TT-tensor by Optima-TT:
        n_opt_bs2, y_opt_bs2 = bs_optima_tt(f, func.n, M)
        print(f'BS2 : {y_opt_bs2:-9.2e}')

        # BS3 OnePlusOne method from nevergrad:
        n_opt_bs3, y_opt_bs3 = bs_nevergrad(f, func.n, M_ng, 'OnePlusOne')
        print(f'BS3 : {y_opt_bs3:-9.2e}')

        # BS4 PSO method from nevergrad:
        n_opt_bs4, y_opt_bs4 = bs_nevergrad(f, func.n, M_ng, 'PSO')
        print(f'BS4 : {y_opt_bs4:-9.2e}')

        # BS5 PSO method from nevergrad:
        n_opt_bs5, y_opt_bs5 = bs_nevergrad(f, func.n, M_ng, 'NoisyBandit')
        print(f'BS5 : {y_opt_bs5:-9.2e}')

        # BS6 PSO method from nevergrad:
        n_opt_bs6, y_opt_bs6 = bs_nevergrad(f, func.n, M_ng, 'SPSA')
        print(f'BS6 : {y_opt_bs6:-9.2e}')

        # BS7 PSO method from nevergrad:
        n_opt_bs7, y_opt_bs7 = bs_nevergrad(f, func.n, M_ng, 'Portfolio')
        print(f'BS7 : {y_opt_bs7:-9.2e}')

        # Present the result:
        text = ''
        text += func.name + ' ' * max(0, 12-len(func.name)) +  ' | '
        text += f'OWN {y_opt_own:-9.2e} | ' #
        text += f'BS1 {y_opt_bs1:-9.2e} | ' # TTOpt
        text += f'BS2 {y_opt_bs2:-9.2e} | ' # Optima-TT
        text += f'BS3 {y_opt_bs3:-9.2e} | ' # nevergrad OnePlusOne
        text += f'BS4 {y_opt_bs4:-9.2e} | ' # nevergrad PSO
        text += f'BS5 {y_opt_bs5:-9.2e} | ' # nevergrad NoisyBandit
        text += f'BS6 {y_opt_bs6:-9.2e} | ' # nevergrad SPSA
        text += f'BS7 {y_opt_bs7:-9.2e} | ' # nevergrad Portfolio
        log(text)


def calc_qubo(d=5, M=1.E+2, M_ng=1.E+4):
    """Perform computations for QUBO problem."""
    M = int(M)
    M_ng = int(M_ng or M)

    nm = 'QUBO'
    log = Log(f'result/logs/qubo.txt')
    log(f'--> {nm} | d={d} | M={M:-7.1e} | Mng={M_ng:-7.1e}')

    for task in [1, 2, 3]:
        f_batch = build_qubo(d, task)
        f = lambda i: f_batch(np.array(i).reshape(1, -1))[0]

        # OWN: Find min value for the original tensor by the proposed method:
        n_opt_own, y_opt_own = protes(f_batch, d, 2, M,
            batch=True, log=True)

        # BS1: Find min value the original tensor by TTOpt:
        n_opt_bs1, y_opt_bs1 = bs_ttopt(f_batch, [2]*d, M)
        print(f'BS1 : {y_opt_bs1:-9.2e}')

        # BS2: Find min value for TT-tensor by Optima-TT:
        n_opt_bs2, y_opt_bs2 = bs_optima_tt(f_batch, [2]*d, M)
        print(f'BS2 : {y_opt_bs2:-9.2e}')

        # BS3 OnePlusOne method from nevergrad:
        n_opt_bs3, y_opt_bs3 = bs_nevergrad(f, [2]*d, M_ng, 'OnePlusOne')
        print(f'BS3 : {y_opt_bs3:-9.2e}')

        # BS4 PSO method from nevergrad:
        n_opt_bs4, y_opt_bs4 = bs_nevergrad(f, [2]*d, M_ng, 'PSO')
        print(f'BS4 : {y_opt_bs4:-9.2e}')

        # BS5 NoisyBandit method from nevergrad:
        n_opt_bs5, y_opt_bs5 = bs_nevergrad(f, [2]*d, M_ng, 'NoisyBandit')
        print(f'BS5 : {y_opt_bs5:-9.2e}')

        # BS6 SPSA method from nevergrad:
        n_opt_bs6, y_opt_bs6 = bs_nevergrad(f, [2]*d, M_ng, 'SPSA')
        print(f'BS6 : {y_opt_bs6:-9.2e}')

        # BS7 Portfolio method from nevergrad:
        n_opt_bs7, y_opt_bs7 = bs_nevergrad(f, [2]*d, M_ng, 'Portfolio')
        print(f'BS7 : {y_opt_bs7:-9.2e}')

        # Present the result:
        text = ''
        text += f' QUBO-{task} | '
        text += f'OWN {y_opt_own:-9.2e} | ' #
        text += f'BS1 {y_opt_bs1:-9.2e} | ' # TTOpt
        text += f'BS2 {y_opt_bs2:-9.2e} | ' # Optima-TT
        text += f'BS3 {y_opt_bs3:-9.2e} | ' # nevergrad OnePlusOne
        text += f'BS4 {y_opt_bs4:-9.2e} | ' # nevergrad PSO
        text += f'BS5 {y_opt_bs5:-9.2e} | ' # nevergrad NoisyBandit
        text += f'BS6 {y_opt_bs6:-9.2e} | ' # nevergrad SPSA
        text += f'BS7 {y_opt_bs7:-9.2e} | ' # nevergrad Portfolio
        log(text)


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'func'

    if mode == 'control':
        calc_control()
    elif mode == 'control_constr':
        calc_control(constr=True)
    elif mode == 'func':
        calc_func()
    elif mode == 'qubo':
        calc_qubo()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
