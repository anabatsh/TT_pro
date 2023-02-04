from .bs_opt_tens_ng import BsOptTensNg


class BsOptTensNgOPO(BsOptTensNg):
    def __init__(self, name='ng_opo'):
        super().__init__(name)
        self.optimizer_class = 'OnePlusOne'
