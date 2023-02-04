from .bs_opt_tens import BsOptTens


try:
    from ttopt import TTOpt
    with_ttopt = True
except Exception as e:
    with_ttopt = False


class BsOptTensTTOpt(BsOptTens):
    def __init__(self, name='ttopt'):
        super().__init__(name)

    def _init(self):
        if not with_ttopt:
            self.err = 'Need "ttopt" module'
            return

    def _optimize(self):
        tto = TTOpt(self.f, d=self.d, n=self.n, evals=self.M, is_func=False)
        tto.minimize()

        self.i_opt = tto.i_min
        self.y_opt = self.f(self.i_opt)
