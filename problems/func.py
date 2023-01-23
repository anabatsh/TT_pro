import teneva


FUNC_LIST = [
    'Ackley', 'Alpine', 'Dixon', 'Exponential',
    'Grienwank', 'Michalewicz', 'Qing', 'Rastrigin', 'Schaffer', 'Schwefel'
]


def func_build_function(d, n, name='Schaffer'):
    func = teneva.func_demo_all(d, names=[name])[0]
    func.set_grid(n, kind='uni')
    return func.get_f_ind
