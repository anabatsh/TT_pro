import numpy as np


from bm_opt_tens import *
from bs_opt_tens import *
from protes import protes
from protes_jax import protes_jax
from dev_protes_jax import dev_protes_jax
from utils import Log
from utils import folder_ensure


# Benchmarks:
bm_all = [
    BmOptTensKnapsack(),
    BmOptTensMaxCut(d=50),
    BmOptTensMvc(d=50),
    BmOptTensKnapsackQuad(d=50),

    BmOptTensControl(d=25),
    BmOptTensControl(d=50),
    BmOptTensControl(d=100),

    BmOptTensControlConstr(d=25),
    BmOptTensControlConstr(d=50),
    BmOptTensControlConstr(d=100),
]


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


def check(m=1.E+4, use_jax=True, with_bs=False, with_log=True):
    log = Log(f'result/logs/check.txt')
    log(f'--> Computations | m={m:-7.1e}')

    for bm in bm_all:
        log('')

        name = f'{bm.name}-{bm.d}D'
        text = name + ' ' * max(0, 20-len(name)) +  ' >>> '
        np.random.seed(42)
        i_opt, y_opt = dev_protes_jax(bm.f, bm.n, m, log=with_log)
        text += f'DEV {y_opt:-9.2e} | '

        np.random.seed(42)
        i_opt, y_opt = protes_jax(bm.f, bm.n, m, log=with_log)
        text += f'JAX {y_opt:-9.2e} | '

        if with_bs:
            for i, bs in enumerate(bs_all, 1):
                bs.prep(bm.f, bm.n, m).optimize()
                text += f'BS{i} {bs.y_opt:-9.2e} | '

        log(text)
        break


if __name__ == '__main__':
    np.random.seed(42)

    folder_ensure('result')
    folder_ensure('result/logs')

    check()
