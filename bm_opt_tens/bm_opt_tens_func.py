import numpy as np


try:
    import teneva
    with_teneva = True
except Exception as e:
    with_teneva = False


from .bm_opt_tens import BmOptTens


class BmOptTensFunc(BmOptTens):
    def __init__(self, d=50, n=10, name='func'):
        # Abstract class
        super().__init__(d, n, name)
        self.desc = ''
        self.name_func = None

    def prep(self, with_shift=True):
        if not with_teneva:
            self.err = 'Need "teneva" module'
            return

        name = self.name.split()
        self.func = teneva.func_demo_all(self.d, names=[self.name_func])[0]
        self.func.set_grid(self.n, kind='uni')
        self.n = self.func.n

        # Translate the function limits to ensure correct competition:
        if with_shift:
            shift = np.random.randn(self.d) / 10
            a_new = self.func.a - (self.func.b-self.func.a) * shift
            b_new = self.func.b + (self.func.b-self.func.a) * shift
            self.func.set_lim(a_new, b_new)

    def _f(self, i):
        return self.func.get_f_ind(i)

    def _f_batch(self, I):
        return self.func.get_f_ind(I)
