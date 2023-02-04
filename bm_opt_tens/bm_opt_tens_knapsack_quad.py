import numpy as np


try:
    import qubogen
    with_qubogen = True
except Exception as e:
    with_qubogen = False


from .bm_opt_tens import BmOptTens


class BmOptTensKnapsackQuad(BmOptTens):
    def __init__(self, d=50, name='knapsack_quad'):
        super().__init__(d, 2, name)
        self.desc = 'QUBO Quadratic Knapsack Problem'
        self.prep()

    def prep(self):
        if not with_qubogen:
            self.err = 'Need "qubogen" module'
            return

        v = np.diag(np.random.random(self.d)) / 3.
        a = np.random.random(self.d)
        b = np.mean(a)
        self.Q = qubogen.qubo_qkp(v, a, b)

    def _f_batch(self, I):
        return ((I @ self.Q) * I).sum(axis=1)
