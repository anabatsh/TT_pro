import numpy as np


from bm_opt_tens import *
from bs_opt_tens import *
from protes_jax import protes_jax
from utils import Log
from utils import folder_ensure


# Baselines:
bs_all = [
    BsOptTensTTOpt(),
    BsOptTensOptimaTT(),
    BsOptTensNgOPO(),
    BsOptTensNgPSO(),
    BsOptTensNgNB(),
    BsOptTensNgSPSA(),
    BsOptTensNgPortfolio(),
]


def check(m=1.E+2, with_bs=False, with_log=True):
    log = Log(f'result/logs/check_topopt.txt')
    log(f'--> Computations | m={m:-7.1e}')

    bm = BmOptTensTopopt()
    log(bm.info())

    name = f'{bm.name}-{bm.d}D'
    text = name + ' ' * max(0, 20-len(name)) +  ' >>> '

    log('\nBaseline solver :')
    i_opt_bm = bm.optimize_bm()
    bm.plot(i_opt_bm, name='Baseline', fpath='result/topopt/topopt_bm.png')
    log('')

    log('\nPROTES solver :')
    i_opt, y_opt = protes_jax(bm.f, bm.n, m, log=with_log)
    bm.plot(i_opt, name='PROTES', fpath='result/topopt/topopt.png')
    text += f'PRO {y_opt:-9.2e} | '

    if with_bs:
        for i, bs in enumerate(bs_all, 1):
            bs.prep(bm.f, bm.n, m).optimize()
            text += f'BS{i} {bs.y_opt:-9.2e} | '

    log(text)


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs')
    folder_ensure('result/topopt')

    check()
