from .bs_opt_tens_ng import BsOptTensNg


class BsOptTensNgNB(BsOptTensNg):
    def __init__(self, name='ng_nb'):
        super().__init__(name)
        self.optimizer_class = 'NoisyBandit'
