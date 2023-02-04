from .bm_opt_tens_func import BmOptTensFunc


class BmOptTensFuncPiston(BmOptTensFunc):
    def __init__(self, d=7, n=10, name='func_piston'):
        super().__init__(d, n, name)
        self.desc = ''
        self.name_func = 'Piston'
        self.prep()

        if d != 7:
            raise ValueError('Dimension should be 7 for Piston function')
