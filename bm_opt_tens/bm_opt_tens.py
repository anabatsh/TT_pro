import numpy as np


class BmOptTens:
    def __init__(self, d, n, name='bm'):
        self.d = d
        self.n = [n] * d if isinstance(n, (int, float)) else n
        self.name = name
        self.desc = ''
        self.opts = {}
        self.err = ''

    def __call__(self, I):
        return self.f(I)

    def f(self, I):
        if self.err:
            raise ValueError(f'Benchmark "{self.name}" is not ready')

        I = np.asanyarray(I, dtype=int)
        if len(I.shape) == 2:
            return self._f_batch(I)
        else:
            return self._f(I)

    def _f(self, i):
        return self._f_batch(np.array(i).reshape(1, -1))[0]

    def _f_batch(self, I):
        return np.array([self._f(i) for i in I])
