from .bs_opt_tens_ng import BsOptTensNg


class BsOptTensNgPSO(BsOptTensNg):
    def __init__(self, name='ng_pso'):
        super().__init__(name)
        self.optimizer_class = 'PSO'
