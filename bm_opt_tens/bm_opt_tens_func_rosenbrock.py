from .bm_opt_tens_func import BmOptTensFunc


class BmOptTensFuncRosenbrock(BmOptTensFunc):
    def __init__(self, d=50, n=10, name='func_'):
        super().__init__(d, n, name)
        self.desc = ''
        self.name_func = 'Rosenbrock'
        self.prep()
