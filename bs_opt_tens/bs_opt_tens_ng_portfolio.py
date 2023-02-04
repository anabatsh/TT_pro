from .bs_opt_tens_ng import BsOptTensNg


class BsOptTensNgPortfolio(BsOptTensNg):
    def __init__(self, name='ng_portfolio'):
        super().__init__(name)
        self.optimizer_class = 'Portfolio'
