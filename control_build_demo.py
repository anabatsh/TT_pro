from gekko import GEKKO
import numpy as np


def control_build_demo(d, x_0=0.8, x_ref=0.7, t_max=1.):
    time = np.linspace(0, t_max, d)

    def ode(x, i):
        # Differential equation:
        return x**3 - i

    def F(x, i):
        # Objective function:
        return 0.5 * ((x - x_ref) ** 2)

    def f(const_i):
        try:
            const_i = list(const_i)

            m = GEKKO(remote=False)
            m.time = time

            x = m.Var(value=x_0, name='x')
            i = m.Param(const_i, name='i')
            obj_val = m.Var(value=0.0)

            m.Equation(x.dt() == ode(x, i))
            m.Equation(obj_val == F(x, i))

            m.options.IMODE = 4
            m.solve(disp=False)
            res = sum(obj_val.VALUE)

        except Exception as e:
            res = 1.E+50

        return res

    return f, (ode, F, time, x_0)


def control_solve_baseline_demo(ode, F, time, x_0, integer=True):
    m = GEKKO(remote=False)
    m.time = time
    m.options.SOLVER = 1 if integer else 3

    x = m.Var(value=x_0, name='x')
    i = m.Var(value=0.0, integer=integer, lb=0, ub=1, name='i')

    m.Equation(x.dt() == ode(x, i))
    m.Obj(F(x, i))

    m.options.IMODE = 6
    m.solve(disp=False)

    return [int(ii) for ii in list(i)], m.options.OBJFCNVAL
