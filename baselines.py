from gekko import GEKKO
import nevergrad as ng
import teneva
from ttopt import TTOpt


def bs_control_gekko(ode, F, time, x_0, constr=False, integer=True):
    m = GEKKO(remote=False)
    m.time = time
    m.options.SOLVER = 1 if integer else 3

    x = m.Var(value=x_0, name='x')
    i = m.Var(value=0.0, integer=integer, lb=0, ub=1, name='i')

    m.Equation(x.dt() == ode(x, i))

    if constr:
        a = m.Var()
        b = m.Var()
        c = m.Var()
        m.delay(i, a, 1)
        m.delay(i, b, 2)
        m.delay(i, c, 3)
        m.Equation(a - b - i <= 0)
        m.Equation(a - c - i <= 0)

    m.Obj(F(x, i))

    m.options.IMODE = 6
    m.solve(disp=False)

    return [int(ii) for ii in list(i)], m.options.OBJFCNVAL


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


def bs_optima_tt(f, N, M, batch=True):
    Y = teneva.tensor_rand(N, r=1)
    Y = teneva.cross(f, Y, e=1.E-16, m=M, dr_max=2)
    Y = teneva.truncate(Y, e=1.E-16)

    n_opt = teneva.optima_tt(Y)[0]
    y_opt = f(n_opt.reshape(1, -1))[0] if batch else f(n_opt)

    return n_opt, y_opt


def bs_ttopt(f, N, M, batch=True):
    tto = TTOpt(f, d=len(N), n=N, evals=M, is_func=False)
    tto.minimize()

    n_opt = tto.i_min
    y_opt = f(n_opt.reshape(1, -1))[0] if batch else f(n_opt)

    return n_opt, y_opt
