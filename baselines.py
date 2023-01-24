import nevergrad as ng
import teneva
from ttopt import TTOpt


def bs_nevergrad(f, N, M, method='PSO'):
    opt_func = eval(f'ng.optimizers.{method}')
    optimizer = opt_func(
        parametrization=ng.p.TransitionChoice(range(N[0]), repetitions=len(N)),
        budget=M, num_workers=1)

    recommendation = optimizer.provide_recommendation()

    for _ in range(optimizer.budget):
        x = optimizer.ask()
        optimizer.tell(x, f(x.value))

    n_opt = optimizer.provide_recommendation().value
    y_opt = f(n_opt)

    return n_opt, y_opt


def bs_optima_tt(f, N, M):
    Y = teneva.tensor_rand(N, r=1)
    Y = teneva.cross(f, Y, e=1.E-16, m=M, dr_max=2)
    Y = teneva.truncate(Y, e=1.E-16)

    n_opt = teneva.optima_tt(Y)[0]
    y_opt = f(n_opt)

    return n_opt, y_opt


def bs_ttopt(f, N, M):
    tto = TTOpt(f, d=len(N), n=N, evals=M, is_func=False)
    tto.minimize()

    n_opt = tto.i_min
    y_opt = f(n_opt)

    return n_opt, y_opt
