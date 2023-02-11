from bm_opt_tens import *
from protes_jax import protes_jax


def demo(m=1.E+2, with_log=True):
    bm = BmOptTensMmul(size=2, rank=7, only2=False)
    # Note: for 2x2 problem bm.f(bm.i_min) is really zero!
    print(bm.info())

    name = f'{bm.name}-{bm.d}D'
    text = name + ' ' * max(0, 20-len(name)) +  ' >>> '

    i_opt, y_opt = protes_jax(bm.f, bm.n, m, log=with_log)
    U, V, W = bm.recover(i_opt)
    text += f'JAX {y_opt:-9.2e} | '
    print(text + '\n\n---> Resulting matrices:\n')
    print(U)
    print(V)
    print(W)


if __name__ == '__main__':
    demo()
