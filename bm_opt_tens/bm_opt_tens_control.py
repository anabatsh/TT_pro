import numpy as np
import os


try:
    from gekko import GEKKO
    with_gekko = True
except Exception as e:
    with_gekko = False


from .bm_opt_tens import BmOptTens


class BmOptTensControl(BmOptTens):
    def __init__(self, d=50, name='control'):
        super().__init__(d, 2, name)
        self.desc = ''
        self.prep()

    def constr(self, const_i):
        return None

    def ode(self, x, i):
        # Differential equation:
        return x**3 - i

    def F(self, x, i):
        # Objective function:
        return 0.5 * ((x - self.x_ref) ** 2)

    def prep(self, x_0=0.8, x_ref=0.7, t_max=1.):
        if not with_gekko:
            self.err = 'Need "gekko" module'
            return

        self.x_0 = x_0
        self.x_ref = x_ref
        self.time = np.linspace(0, t_max, self.d)
        self.opts = (self.ode, self.F, self.time, self.x_0)

    def _f(self, const_i):
        const_i = list(const_i)

        constr = self.constr(const_i)
        if constr is not None:
            return constr

        try:
            m = GEKKO(remote=False)
            m.time = self.time

            x = m.Var(value=self.x_0, name='x')
            i = m.Param(const_i, name='i')
            obj_val = m.Var(value=0.0)

            m.Equation(x.dt() == self.ode(x, i))
            m.Equation(obj_val == self.F(x, i))

            m.options.IMODE = 4
            m.solve(disp=False)
            res = sum(obj_val.VALUE)

        except Exception as e:
            res = 1.E+50
        finally:
            os.system('find /tmp -maxdepth 1 -type d  -user $USER  -name "*model*" -exec rm -fr {} >/dev/null 2>/dev/null \;')

        return res
