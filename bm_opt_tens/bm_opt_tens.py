import numpy as np


class BmOptTens:
    def __init__(self, d, n, name='bm'):
        self.d = d
        self.n = [n] * d if isinstance(n, (int, float)) else n
        self.name = name
        self.desc = ''
        self.opts = {}
        self.err = ''

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
        text += self.name + ' ' * max(0, 20-len(self.name)) +  ' | '
        text += f'DIM = {self.d}:-4d | <MODE> = {np.mean(self.n):-4.1f}\n'
        if self.desc:
            desc = f'  [ {self.desc.strip()} ]'
            text += desc.replace('            ', '    ')
        text += '\n' + '=' * 72 + '\n'
        return text
