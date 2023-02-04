from .bm_opt_tens_func import BmOptTensFunc


class BmOptTensFuncSchwefel(BmOptTensFunc):
    def __init__(self, d=50, n=10, name='func_schwefel'):
        super().__init__(d, n, name)
        self.desc = ''
        self.name_func = 'Schwefel'
        self.prep()
