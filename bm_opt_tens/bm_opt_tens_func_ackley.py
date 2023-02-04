from .bm_opt_tens_func import BmOptTensFunc


class BmOptTensFuncAckley(BmOptTensFunc):
    def __init__(self, d=50, n=10, name='func_ackley'):
        super().__init__(d, n, name)
        self.desc = ''
        self.name_func = 'Ackley'
        self.prep()
