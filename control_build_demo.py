from gekko import GEKKO
import numpy as np


def control_build_demo(d, x_0=0.8, x_ref=0.7, t_max=1., constr=False):
    time = np.linspace(0, t_max, d)

    def ode(x, i):
        # Differential equation:
        return x**3 - i

    def F(x, i):
        # Objective function:
        return 0.5 * ((x - x_ref) ** 2)

    def f(const_i):
        const_i = list(const_i)
        if constr:
            const_s = "".join([str(i) for i in const_i])
            if const_s.startswith('10') or const_s.startswith('110') or \
                const_s.endswith('01') or const_s.endswith('011') or \
                '010' in const_s or '0110' in const_s:
                    return 1e+50
        try:

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


def control_solve_baseline_demo(ode, F, time, x_0, integer=True, constr=False):
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
