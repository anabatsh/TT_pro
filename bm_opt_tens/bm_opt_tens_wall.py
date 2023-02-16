import numpy as np


from .bm_opt_tens import BmOptTens


class BmOptTensWall(BmOptTens):
    def __init__(self, d=10, n=50, name='wall'):
        super().__init__(d, n, name)
        self.desc = 'Tensor of the special form'

        self.i_min = np.zeros(self.d)

    def _f(self, i):
        if len(np.where(i == self.i_min)[0]) == self.d:
            return 0.
        elif len(np.where(i == self.i_min)[0]) > 0:
            return self.d * 10
        else:
            return i[0]
