import teneva
from ttopt import TTOpt


def func_run(d, n, M, opt, log):
    for func in teneva.func_demo_all(d):
        # Set the grid:
        func.set_grid(n, kind='uni')

        # OWN: Find min value for the original tensor by the proposed method:
        n_opt_own = opt(func.get_f_ind)
        y_opt_own =func.get_f_ind(n_opt_own)

        # BS1: Find min value the original tensor by TTOpt:
        tto = TTOpt(func.get_f_ind, d=d, n=n, evals=M, is_func=False)
        tto.minimize()
        n_opt_bs1 = tto.i_min
        y_opt_bs1 = func.get_f_ind(n_opt_bs1)

        # BS2: Find min value for TT-tensor by Optima-TT
        # (we build the TT-approximation by the TT-CROSS method):
        Y = teneva.tensor_rand(func.n, r=1)
        Y = teneva.cross(func.get_f_ind, Y, e=1.E-16, m=M, dr_max=2)
        Y = teneva.truncate(Y, e=1.E-16)
        n_opt_bs2 = teneva.optima_tt(Y)[0]
        y_opt_bs2 = func.get_f_ind(n_opt_bs2)

        # Present the result:
        text = ''
        text += func.name + ' ' * max(0, 12-len(func.name)) +  ' | '
        text += f'PROTES: {y_opt_own:-9.2e} | '
        text += f'TTOpt: {y_opt_bs1:-9.2e} | '
        text += f'Optima-TT: {y_opt_bs2:-9.2e} | '
        log(text)
