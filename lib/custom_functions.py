import numpy as np
from .show_functions import show_X
from .gekko_functions import optimal_control_substituting


def sample(n_samples, probabilities, P=None, max_sample_steps=10):
    """
    Sample K binary vectors from q(p_1, ..., p_{T-1})
    according to the constraint P
    params: 
        n_samples = K - number of samples
        probabilities - parameters p_1 ... p_{T-1}
        P - constraint boolean function - if None there is no constraint
        max_sample_steps - max number of resampling
    returns:
        I - samples
    """
    T = len(probabilities)
    
    if P is None:
        I = (np.random.rand(n_samples, T) <= probabilities).astype(np.int32)
        return I
    
    mask = np.ones(n_samples).astype(np.bool_)
    I = np.zeros((n_samples, T)).astype(np.int32)
    print('-'*50)
    print('Start sampling')
    for _ in range(max_sample_steps):
        I[mask] = (np.random.rand(mask.sum(), T) <= probabilities).astype(np.int32)
        mask = ~P(I)
        print(f'Wrong {mask.mean()*100:<3}%')
        if (~mask).all():
            break
#     print(f'Early stop: {_}')
    return I

def indicatr(x, lamb):
    """
    Indicator function of the top-s best (minimal) values:
    If x_i from x is one of the s minimal values of x, we 
    return 1 for this element and 0 otherwise
    params: 
        x - 1D real array 
        lamb - proportion of the best values s.t s = lamb% of x
    returns:
        weights - binary vector with the indicator function values
    """
    top_s = int(lamb * len(x))
    w = (x <= np.sort(x)[top_s-1]).astype(np.float64)
    return w

def exponent(x, loc=0.0, lamb=1.0):
    """
    Exponent probability distribution function
    params: 
        x - 1D real array 
        lamb - parameter of the distribution
    returns:
        weights - real vector with the corresponding pdf values
    """
    x_loc = x - loc
    w = np.where(x_loc >= 0, lamb * np.exp(-lamb * x_loc), 0)
    return w

def optimal_control_custom(
    x_0, x_ref, time, f, F, 
    n_samples=10, P=None, max_sample_steps=10, n_steps=10, 
    mode='top-s', verbose=0, update=True, **kwargs
):
    """
    params:
        x_0 - initial state
        x_ref - final reference state  
        time - time horizon
        f - differential equation
        F - objective function
        n_samples = K - number of samples
        P - constraint boolean function - if None there is no constraint
        max_sample_steps - max number of resampling
        mode - way of calculating weights of the samples
            'top-s' : kwargs.lamb - proportion of the best solutions
            'expon' : kwargs.lamb - parameter of the exponential distribution
        n_steps = N - number of steps
        verbose - output setting
            0 - show solutions
            1 - print objective values only
        update - True if the best solution is remembered
    returns:
        history: a dict with
            x - predicted state trjectory of x(t)
            i - given sequense of controls i(t)
            obj_val - corresponding objective value F(x, i)
            x_0, x_ref, time and some other items
    """
    best_sol = {'step' : 0, 'obj_val' : np.inf, 'x': None, 'i': None}
    
    p = np.ones_like(time) * 0.5
    history = {'p_trace' : [p], 'obj_val_trace': []}

    if mode == 'top-s':
        w_func = indicatr
    else:
        w_func = exponent
    
    # main iteration loop
    for step in range(1, n_steps+1):
        X = []
        Obj = []
        
        # sample and substitute solutions
        I = sample(n_samples, p, P, max_sample_steps)
        for const_i in I:
            history_i = optimal_control_substituting(x_0, x_ref, time, f, F, const_i)
            X.append(history_i['x'])
            Obj.append(history_i['obj_val'])
        Obj = np.array(Obj)
        
        # remember the best current solution
        min_id = np.argmin(Obj)
        curr_sol = {'step' : step, 'obj_val' : Obj[min_id], 'x': X[min_id], 'i': I[min_id]}
        
        # update the best total solution if specified
        if update and curr_sol['obj_val'] < best_sol['obj_val'] or not update:
            best_sol = curr_sol
            
        # compute the weights of the solutions
        w = w_func(Obj, **kwargs)
        
        # show the intermediate results
        if verbose == 0 or verbose == 1:
            print('-'*80)
            c, b = curr_sol['obj_val'], best_sol['obj_val']
            print(f'Step {step}/{n_steps} | Objective Value (current) {c:.5f} | Objective Value (total) {b:.5f}')
            if verbose == 0:
                show_X(X, x_0, x_ref, time, w, p)

        # update the probabilities
        p = np.average(I, axis=0, weights=w)
        
        history['obj_val_trace'].append(Obj)
        history['p_trace'].append(p)
        
    history['time'] = time
    history['x'] = best_sol['x']
    history['i'] = best_sol['i']
    history['obj_val'] = best_sol['obj_val']
    history['x_0'] = x_0
    history['x_ref'] = x_ref
    history['obj_val_trace'] = np.array(history['obj_val_trace'])
    history['p_trace'] = np.array(history['p_trace'])
    return history