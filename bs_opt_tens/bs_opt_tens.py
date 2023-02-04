import numpy as np
from time import perf_counter as tpc


class BsOptTens:
    def __init__(self, name='bs'):
        self.name = name
        self.err = ''
        self.is_prep = False

        self.t = 0.
        self.i_opt = None
        self.y_opt = None

        self._init()

    def prep(self, f=None, n=[1], M=1.E+4):
        self.f = f
        self.d = len(n)
        self.n = n
        self.M = int(M)

        self.is_prep = True

        return self

    def optimize(self):
        if not self.is_prep:
            self.err = 'Call "prep" method for baseline before usage'

        if self.err:
            raise ValueError(f'Baseline {self.name} is not ready')

        _t = tpc()
        self._optimize()
        self.t = tpc() - _t

    def _init(self):
        return

    def _optimize(self):
        raise NotImplementedError()
