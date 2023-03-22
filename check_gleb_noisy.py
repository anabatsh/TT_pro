import numpy as np
import sys, os


from bm_opt_tens import *
from bs_opt_tens import *
# from protes import protes
from protes_jax import protes_jax
from dev3_protes_jax import protes_jax as protes_gleb
from dev4_protes_jax_rej import protes_jax_rej as protes_gleb_rej
from protes1r import protes1r
from utils import Log
from utils import folder_ensure


# Benchmarks:
bm_all = [
    BmOptTensFuncAckley(d=7, n=10),
    BmOptTensFuncAlpine(d=7, n=10),
    # BmOptTensFuncDixon(d=7, n=10),
    BmOptTensFuncExp(d=7, n=10),
    BmOptTensFuncGrienwank(d=7, n=10),
    BmOptTensFuncMichalewicz(d=7, n=10),
    BmOptTensFuncPiston(n=10),
    BmOptTensFuncQing(d=7, n=10),
    BmOptTensFuncRastrigin(d=7, n=10),
    # BmOptTensFuncRosenbrock(d=7, n=10),
    BmOptTensFuncSchaffer(d=7, n=10),
    BmOptTensFuncSchwefel(d=7, n=10),

    BmOptTensMaxCut(d=50),
    BmOptTensMvc(d=50),
    BmOptTensKnapsackQuad(d=50),
    BmOptTensKnapsack(),

    BmOptTensControl(d=25),
    BmOptTensControl(d=50),
    BmOptTensControl(d=100),

    BmOptTensControlConstr(d=25),
    BmOptTensControlConstr(d=50),
    BmOptTensControlConstr(d=100),
]

bm_all = [

    #BmOptTensMaxCut(d=50),
    #BmOptTensMvc(d=50),
    #BmOptTensKnapsackQuad(d=50),
    #BmOptTensKnapsack(),


    # BmOptTensMmul(size=2, rank=7, only2=False, E = [-1, 0, 1])


    # BmOptTensMaxCut(d=150),
    # BmOptTensMvc(d=150),
    # BmOptTensKnapsackQuad(d=150),
    BmOptTensControl(d=25),
    BmOptTensControl(d=50),
    BmOptTensControl(d=100),
    BmOptTensControl(d=200),
]

# Baselines:
bs_all = [
]


def check(M=1.E+4, *, r=2, k_gd=100, how_to_upd=True, T=0.5, use_jax=True, with_bs=True, with_log=True, compare=False):
    log = Log(f'result/logs/check_orth{os.getpid()}.txt')
    log(f'--> Computations | M={M:-7.1e}')

    for bm in bm_all:
        log('')

        name = f'{bm.name}-{bm.d}D'
        text = name + ' ' * max(0, 20-len(name)) +  ' >>> '

        info = dict()
        opt = protes_gleb_rej
        i_opt, y_opt = opt(bm.f, bm.n, M, log=with_log, info=info, T=T, r=r, k_gd=k_gd, how_to_upd=how_to_upd)
        text += f'GRY {y_opt:-9.2e} needed M: {info["m_opt_list"]} | '

        if compare:
            info = dict()
            opt = protes_gleb
            i_opt, y_opt = opt(bm.f, bm.n, M, log=with_log, info=info, r=r)
            text += f'GMY {y_opt:-9.2e} needed M: {info["m_opt_list"]} | '

            info = dict()
            opt = protes_jax if use_jax else protes
            i_opt, y_opt = opt(bm.f, bm.n, M, log=with_log, info=info, r=r)
            text += f'OWN {y_opt:-9.2e} needed M: {info["m_opt_list"]} | '

            if with_bs:
                for i, bs in enumerate(bs_all, 1):
                    bs.prep(bm.f, bm.n, M).optimize()
                    text += f'BS{i} {bs.y_opt:-9.2e} | '

        log(text)


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs')

    print(sys.argv[1:])
    try:
        r = int(sys.argv[1])
    except:
        r = 2

    try:
        k_gd = int(sys.argv[2])
    except:
        k_gd = 20

    try:
        htu = sys.argv[3][0] == 'T'
    except:
        htu = True

    try:
        T = float(sys.argv[4])
    except:
        T = 0.5

    print(r, k_gd, htu, T)
    test = True

    if test:
        check(compare=True, r=r, k_gd=k_gd, how_to_upd=htu, T=T)
    else:
        bm_all = [BmOptTensMmul(size=2, rank=7, only2=True, E = [-1, 0, 1])]
        check(M=1e8, r=r, k_gd=k_gd, how_to_upd=htu, T=T)



