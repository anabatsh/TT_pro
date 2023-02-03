import numpy as np
from time import perf_counter as tpc


from protes import protes
from build_control import build_control
from build_knapsack import build_knapsack
from utils import Log
from utils import folder_ensure


def calc(M=int(1.E+2), r_list=[1, 3], reps=1):
    log = Log(f'result/logs_check_rank/result.txt')

    d, f = build_knapsack()

    for r in r_list:
        name = f'antenna-{d}D-{r}R'
        text = name + ' ' * max(0, 15-len(name)) +  ' > '
        y_opt_all = []
        for rep in range(reps):
            t = tpc()
            n_opt, y_opt = protes(f, d, 2, M, r=r, log=True)
            y_opt_all.append(y_opt)
            t = tpc() - t
            if rep == 0:
                text += f't = {t:-9.2e} | '
            text += f'v{rep} = {y_opt:-9.2e} | '
        text += f'<v> = {np.mean(y_opt_all):-9.2e} | '
        log(text)

    d = 5
    constr = False
    f, args = build_control(d, constr=constr)

    for r in r_list:
        name = f'control-{d}D-{r}R'
        text = name + ' ' * max(0, 15-len(name)) +  ' > '
        y_opt_all = []
        for rep in range(reps):
            t = tpc()
            n_opt, y_opt = protes(f, d, 2, M, r=r, constr=constr, log=True)
            y_opt_all.append(y_opt)
            t = tpc() - t
            if rep == 0:
                text += f't = {t:-9.2e} | '
            text += f'v{rep} = {y_opt:-9.2e} | '
        text += f'<v> = {np.mean(y_opt_all):-9.2e} | '
        log(text)


    d = 5
    constr = True
    f, args = build_control(d, constr=constr)

    for r in r_list:
        name = f'control-c-{d}D-{r}R'
        text = name + ' ' * max(0, 15-len(name)) +  ' > '
        y_opt_all = []
        for rep in range(reps):
            t = tpc()
            n_opt, y_opt = protes(f, d, 2, M, r=r, constr=constr, log=True)
            y_opt_all.append(y_opt)
            t = tpc() - t
            if rep == 0:
                text += f't = {t:-9.2e} | '
            text += f'v{rep} = {y_opt:-9.2e} | '
        text += f'<v> = {np.mean(y_opt_all):-9.2e} | '
        log(text)


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs_check_rank')
    calc()
