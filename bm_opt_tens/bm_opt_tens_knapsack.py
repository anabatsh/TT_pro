import numpy as np


from .bm_opt_tens import BmOptTens


class BmOptTensKnapsack(BmOptTens):
    def __init__(self, d=50, name='knapsack'):
        super().__init__(50, 2, name)
        self.desc = 'Binary knapsack problem with fixed weights wi in [5, 20], profits pi in [50, 100] (i = 1, 2, . . . , d) and the maximum capacity C = 1000. It is from work (Dong et al., 2021) (problem k3; d = 50), where anglemodulated bat algorithm (AMBA) algorithm was proposed for high-dimensional binary optimization problems with engineering application to antenna topology optimization.'

        self.prep()

        if d != 50:
            raise ValueError('Dimension should be 50 for Knapsack')

    def prep(self):
        # w = [40, 27, 5, 21, 51, 16, 42, 18, 52, 28, 57, 34, 44, 43, 52, 55, 53,
        # 42, 47, 56, 57, 44, 16, 2, 12, 9, 40, 23, 56, 3, 39,16, 54, 36, 52, 5, 53,
        # 48, 23, 47, 41, 49, 22, 42, 10, 16, 53, 58, 40, 1, 43, 56, 40, 32, 44, 35,
        # 37, 45, 52, 56, 40, 2, 23,49, 50, 26, 11, 35, 32, 34, 58, 6, 52, 26, 31,
        # 23, 4, 52, 53, 19]
        # p = [199, 194, 193, 191, 189, 178, 174, 169, 164, 164, 161, 158, 157,
        # 154, 152, 152, 149, 142, 131, 125, 124, 124, 124, 122, 119, 116, 114,
        # 113, 111, 110, 109, 100, 97, 94, 91, 82, 82, 81, 80, 80, 80, 79, 77, 76,
        # 74, 72, 71, 70, 69,68, 65, 65, 61, 56, 55, 54, 53, 47, 47, 46, 41, 36, 34,
        # 32, 32,30, 29, 29, 26, 25, 23, 22, 20, 11, 10, 9, 5, 4, 3, 1]
        # C = 1173.0

        self.w = [
            80, 82, 85, 70, 72, 70, 66, 50, 55, 25, 50, 55, 40, 48, 59, 32, 22,
            60, 30, 32, 40, 38, 35, 32, 25, 28, 30, 22, 50, 30, 45, 30, 60, 50,
            20, 65, 20, 25, 30, 10, 20, 25, 15, 10, 10, 10, 4, 4, 2, 1]

        self.p = [
            220, 208, 198, 192, 180, 180, 165, 162, 160, 158, 155, 130, 125,
            122, 120, 118, 115, 110, 105, 101, 100, 100, 98, 96, 95, 90, 88, 82,
            80, 77, 75, 73, 72, 70, 69, 66, 65, 63, 60, 58, 56, 50, 30, 20, 15,
            10, 8, 5, 3, 1]

        self.C = 1000

    def _f(self, i):
        cost = np.dot(self.p, i)
        constr = np.dot(self.w, i)
        return 0 if constr > self.C else -cost
