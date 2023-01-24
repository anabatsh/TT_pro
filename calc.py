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
from control_build import control_build
from utils import Log
from utils import folder_ensure


def calc_control(d=100, M=1.E+6, K=20, k=1, k_gd=50, r=5, lr=1.E-4, constr=False):
    log = Log(f'result/logs/control.txt')

    txt = f'--> control | '
    txt += f'd={d} M={M:-7.1e} K={K} k={k} k_gd={k_gd} r={r}'
    log(txt)

    f, args = control_build(d, constr=constr)

    M = int(M)

    def opt(f):
        return protes(f, d, 2, M, K, k, k_gd, r, lr, batch=False, log=True, constr=constr)

    # OWN: Find min value for the original tensor by the proposed method:
    n_opt_own, y_opt_own = opt(f)

    # BS1: Find min value the original tensor by TTOpt:
    n_opt_bs1, y_opt_bs1 = bs_control_gekko(*args)

    # BS2: ???:
    n_opt_bs2, y_opt_bs2 = None, -1

    # BS3 OnePlusOne method from nevergrad:
    n_opt_bs3, y_opt_bs3 = bs_nevergrad(f, [2]*d, M, 'OnePlusOne')

    # BS4 PSO method from nevergrad:
    n_opt_bs4, y_opt_bs4 = bs_nevergrad(f, [2]*d, M, 'PSO')

    # BS5 PSO method from nevergrad:
    n_opt_bs5, y_opt_bs5 = bs_nevergrad(f, [2]*d, M, 'NoisyBandit')

    # BS6 PSO method from nevergrad:
    n_opt_bs6, y_opt_bs6 = bs_nevergrad(f, [2]*d, M, 'SPSA')

    # BS7 PSO method from nevergrad:
    n_opt_bs7, y_opt_bs7 = bs_nevergrad(f, [2]*d, M, 'Portfolio')

    # Present the result:
    text = ''
    text += ' ODE-1 | '
    text += f'OWN {y_opt_own:-9.2e} | ' #
    text += f'BS1 {y_opt_bs1:-9.2e} | ' # GEKKO Base method
    text += f'BS2 {y_opt_bs2:-9.2e} | ' # ???
    text += f'BS3 {y_opt_bs3:-9.2e} | ' # nevergrad OnePlusOne
    text += f'BS4 {y_opt_bs4:-9.2e} | ' # nevergrad PSO
    text += f'BS5 {y_opt_bs5:-9.2e} | ' # nevergrad NoisyBandit
    text += f'BS6 {y_opt_bs6:-9.2e} | ' # nevergrad SPSA
    text += f'BS7 {y_opt_bs7:-9.2e} | ' # nevergrad Portfolio
    log(text)


def calc_control_demo(d=20, M=1.E+3, K=20, k=1, k_gd=50, r=5, lr=1.E-4, constr=False):
    """Solve the optimal control problem (demo)."""
    from control_build_demo import control_build_demo
    from control_build_demo import control_solve_baseline_demo

    log = Log(f'result/logs/control_demo.txt')

    txt = f'--> control_demo | '
    txt += f'd={d} M={M:-7.1e} K={K} k={k} k_gd={k_gd} r={r}'
    log(txt)

    f, opts = control_build_demo(d, constr=constr)

    t = tpc()
    n_opt, y_opt = protes(f, d, 2, M, K, k, k_gd, r, lr,
        batch=False, log=True, log_ind=True)
    t = tpc() - t

    t_ref = tpc()
    n_opt_ref, y_opt_ref = control_solve_baseline_demo(*opts, constr=constr)
    t_ref = tpc() - t_ref

    log(f'\n--------')
    log(f'Result : {y_opt:-14.7e} | Baseline: {y_opt_ref:-14.7e}')
    log(f'Time   : {t:-14.3f} | Baseline: {t_ref:-14.7e}')

    log(f'n opt     >> {"".join([str(n) for n in n_opt])}')
    log(f'n opt ref >> {"".join([str(n) for n in n_opt_ref])}')


def calc_func(d=100, n=64, M=1.E+3, K=20, k=5, k_gd=50, r=5, lr=1.E-4):
    """Perform computations for analytical multivariable functions."""
    log = Log(f'result/logs/func.txt')

    txt = f'--> FUNC | '
    txt += f'd={d} n={n} M={M:-7.1e} K={K} k={k} k_gd={k_gd} r={r}'
    log(txt)

    M = int(M)

    def opt(f):
        return protes(f, d, n, M, K, k, k_gd, r, lr, batch=True,
            with_qtt=True, log=False)

    funcs = teneva.func_demo_all(d, only_with_min=True, only_with_min_x=True)

    for func in funcs:
        # Set the grid:
        func.set_grid(n, kind='uni')

        # Translate the function limits to ensure correct competition:
        shift = 0.32
        a_new = func.x_min - (func.b-func.a) * shift
        b_new = func.x_min + (func.b-func.a) * (1. - shift)
        func.set_lim(a_new, b_new)

        # Target function for optimization:
        f = func.get_f_ind

        # OWN: Find min value for the original tensor by the proposed method:
        n_opt_own, y_opt_own = opt(f)

        # BS1: Find min value the original tensor by TTOpt:
        n_opt_bs1, y_opt_bs1 = bs_ttopt(f, func.n, M)

        # BS2: Find min value for TT-tensor by Optima-TT
        # (we build the TT-approximation by the TT-CROSS method):
        n_opt_bs2, y_opt_bs2 = bs_optima_tt(f, func.n, M)

        # BS3 OnePlusOne method from nevergrad:
        n_opt_bs3, y_opt_bs3 = bs_nevergrad(f, func.n, M, 'OnePlusOne')

        # BS4 PSO method from nevergrad:
        n_opt_bs4, y_opt_bs4 = bs_nevergrad(f, func.n, M, 'PSO')

        # BS5 PSO method from nevergrad:
        n_opt_bs5, y_opt_bs5 = bs_nevergrad(f, func.n, M, 'NoisyBandit')

        # BS6 PSO method from nevergrad:
        n_opt_bs6, y_opt_bs6 = bs_nevergrad(f, func.n, M, 'SPSA')

        # BS7 PSO method from nevergrad:
        n_opt_bs7, y_opt_bs7 = bs_nevergrad(f, func.n, M, 'Portfolio')

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


def calc_qubo(d=250, M=1.E+5, K=20, k=1, k_gd=50, r=5, lr=1.E-4):
    """Solve the QUBO problem."""
    from qubo import qubo_build_function
    from qubo import qubo_build_matrix
    from qubo import qubo_solve_baseline

    log = Log(f'result/logs/qubo.txt')

    txt = f'--> QUBO | '
    txt += f'd={d} M={M:-7.1e} K={K} k={k} k_gd={k_gd} r={r}'
    log(txt)

    Q = qubo_build_matrix(d, prob_con=0.5)

    t_ref = tpc()
    y_opt_ref = qubo_solve_baseline(Q)
    t_ref = tpc() - t_ref

    t = tpc()
    f = qubo_build_function(jnp.array(Q))
    n_opt, y_opt = protes(f, d, 2, M, K, k, k_gd, r, lr,
        batch=True, log=True)
    t = tpc() - t

    log(f'\n--------')
    log(f'Result : {y_opt:-14.7e} | Baseline: {y_opt_ref:-14.7e}')
    log(f'Time   : {t:-14.3f} | Baseline: {t_ref:-14.7e}')


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'func'

    if mode == 'control_cstrn':
        calc_control(constr=True)
    elif mode == 'control_demo_cstrn':
        calc_control_demo(constr=True)
    elif mode == 'control':
        calc_control()
    elif mode == 'control_demo':
        calc_control_demo()
    elif mode == 'func':
        calc_func()
    elif mode == 'qubo':
        calc_qubo()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
