import numpy as np


class BmOptTens:
    def __init__(self, d, n, name='bm'):
        self.d = d
        self.n = [n] * d if isinstance(n, (int, float)) else n
        self.name = name
        self.desc = ''
        self.opts = {}
        self.err = ''

        self.i_min = None
        self.y_min = None
        self.i_max = None
        self.y_max = None

    def __call__(self, I):
        return self.f(I)

    def f(self, I):
        if self.err:
            raise ValueError(f'Benchmark "{self.name}" not ready ({self.err})')

        I = np.asanyarray(I, dtype=int) # TODO! Add jax and torch versions

        if len(I.shape) == 2:
            return self._f_batch(I)
        else:
            return self._f(I)

    def _f(self, i):
        return self._f_batch(np.array(i).reshape(1, -1))[0]

    def _f_batch(self, I):
        return np.array([self._f(i) for i in I])

    def info(self):
        text = '-' * 72 + '\n' + 'BM: '
        text += self.name + ' ' * max(0, 30-len(self.name)) +  ' | '
        text += f'DIMS = {self.d:-5d} | <MODE SIZE> = {np.mean(self.n):-6.1f}\n'

        if self.y_min is not None or self.y_max is not None:
            text += ' ' * 30
            if self.y_min is not None:
                text += f'y_min = {self.y_min:-11.4e} | '
            if self.y_max is not None:
                text += f'y_max = {self.y_max:-11.4e}'
            text += '\n'

        if self.desc:
            desc = f'  [ {self.desc.strip()} ]'
            text += desc.replace('            ', '    ')

        text += '\n' + '=' * 72 + '\n'
        return text
