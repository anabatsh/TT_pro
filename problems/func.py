import sys
import teneva


sys.path.append('.')
from baselines import bs_nevergrad
from baselines import bs_optima_tt
from baselines import bs_ttopt


def func_run(d, n, M, opt, log):
    for func in teneva.func_demo_all(d):
        # Set the grid:
        func.set_grid(n, kind='uni')

        # Translate the function limits to ensure correct competition:
        func.set_lim(a=func.a*0.95, b=func.b*0.97)

        # Target function for optimization:
        f = func.get_f_ind

        # OWN: Find min value for the original tensor by the proposed method:
        n_opt_own = opt(f)
        y_opt_own = f(n_opt_own)

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
