import numpy as np
import os


class Log:
    def __init__(self, fpath=None):
        self.fpath = fpath
        self.is_new = True
        self.len_pref = 10

    def __call__(self, text):
        print(text)
        if self.fpath:
            with open(self.fpath, 'w' if self.is_new else 'a') as f:
                f.write(text + '\n')
        self.is_new = False

    def name(self, name):
        text = '>>> ' + name
        self(text)

    def res(self, res):
        text = ''
        name = res['method']
        text += '  - ' + name + ' ' * max(0, self.len_pref-len(name))
        text += f' | error : {res["e_tst"]:-7.1e}'
        text += f' | time : {res["t"]:-7.3f}'
        self(text)


def folder_ensure(fpath):
    os.makedirs(fpath, exist_ok=True)
