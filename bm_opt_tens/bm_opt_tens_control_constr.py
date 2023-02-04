from .bm_opt_tens_control import BmOptTensControl


class BmOptTensControlConstr(BmOptTensControl):
    def __init__(self, d=50, name='control_constr'):
        super().__init__(d, name)

    def constr(self, const_i):
        const_s = ''.join([str(i) for i in const_i])

        if const_s.startswith('10') or const_s.startswith('110') or \
            const_s.endswith('01') or const_s.endswith('011') or \
            '010' in const_s or '0110' in const_s:
                return 1e+42
