import numpy as np
from time import perf_counter as tpc


from protes import protes
from build_control import build_control
from build_knapsack import build_knapsack
from utils import Log
from utils import folder_ensure


def calc(M=int(1.E+4), r_list=[1, 3, 5, 7, 9], reps=1):
    log = Log(f'result/logs_check_rank/result.txt')

    tasks = {
        'anten': {'d': 50, 'n': 2,
            'f': build_knapsack()[1] },
        'con-b': {'d': 50, 'n': 2,
            'f': build_control(50)[0] },
        'con-c': {'d': 50, 'n': 2,
            'f': build_control(50, constr=True)[0], 'constr': True},
    }

    for name_task, task in tasks.items():
        for r in r_list:
            text = f'{name_task}-{task["d"]}D-{r}R > '
            y_opt_all = []
            for rep in range(reps):
                t = tpc()
                n_opt, y_opt = protes(task['f'], task['d'], task['n'], M, r=r,
                    constr=task.get('constr'), log=True)
                y_opt_all.append(y_opt)
                t = tpc() - t
                if rep == 0:
                    text += f't {t:-7.1e} | v '
                text += f'{y_opt:-8.1e} '
            text += f'| <v> {np.mean(y_opt_all):-9.2e}'
            log(text)


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs_check_rank')

    calc()
