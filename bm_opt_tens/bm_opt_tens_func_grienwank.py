from .bm_opt_tens_func import BmOptTensFunc


class BmOptTensFuncGrienwank(BmOptTensFunc):
    def __init__(self, d=50, n=10, name='func_grienwank'):
        super().__init__(d, n, name)
        self.desc = ''
        self.name_func = 'Grienwank'
        self.prep()
