from .bs_opt_tens_ng import BsOptTensNg


class BsOptTensNgSPSA(BsOptTensNg):
    def __init__(self, name='ng_spsa'):
        super().__init__(name)
        self.optimizer_class = 'SPSA'
