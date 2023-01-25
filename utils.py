import jax.numpy as jnp
import numpy as np
import os
from construct_TT import tens
import teneva

def gen_func_pair(num_ones=3):
    def f0(x):
        if x == 0 or x == num_ones:
            return 0
        
    def f1(x):
        return min(num_ones, x + 1)
    
    return [f0, f1]

def gen_func_pair_last(num_ones=3):
    def f0(x):
        if x == 0 or x == num_ones:
            return 1
        
    def f1(x):
        if x >= num_ones - 1:
            return 1
    
    return [f0, f1]

def ind_tens_max_ones(d, num_ones):
    funcs = [gen_func_pair(num_ones)]*(d-1) +  [gen_func_pair_last(num_ones)]
    cores = tens(funcs).cores
    cores = teneva.orthogonalize(cores, k=0)
    return cores


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


def folder_ensure(fpath):
    os.makedirs(fpath, exist_ok=True)


def get_many(Y, I):
    """Compute the elements of the TT-tensor on many indices."""
    Q = Y[0][0, I[:, 0], :]
    for i in range(1, len(Y)):
        Q = jnp.einsum('kq,qkp->kp', Q, Y[i][:, I[:, i], :])
    return Q[:, 0]
