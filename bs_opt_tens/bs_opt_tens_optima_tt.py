from .bs_opt_tens import BsOptTens


try:
    import teneva
    with_teneva = True
except Exception as e:
    with_teneva = False


class BsOptTensOptimaTT(BsOptTens):
    def __init__(self, name='optimatt'):
        super().__init__(name)

    def _init(self):
        if not with_teneva:
            self.err = 'Need "teneva" module'
            return

    def _optimize(self):
        Y = teneva.tensor_rand(self.n, r=1)
        Y = teneva.cross(self.f, Y, e=1.E-16, m=self.M, dr_max=2)
        Y = teneva.truncate(Y, e=1.E-16)

        self.i_opt = teneva.optima_tt(Y)[0]
        self.y_opt = self.f(self.i_opt)
