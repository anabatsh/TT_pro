import numpy as np
from gekko import GEKKO


def optimal_control_minimization(x_0, x_ref, time, f, F, integer=False):
    """
    Solve the problem with GEKKO
    .---------------------------------------.
    | F -> min s.t.  x(0) = x_0             |
    |                x(t+1) = f(x(t), i(t)) |
    .---------------------------------------.          
    params:
        x_0 - initial state
        x_ref - final reference state  
        time - time horizon
        f - differential equation
        F - objective function
        integer - control integrality flag, default False
    returns: 
        history: a dict with
            x - predicted state trjectory of x(t)
            i - best sequense of controls i(t)
            obj_val - corresponding objective value F(x, i)
            x_0, x_ref, time
    """
    m = GEKKO(remote=False)
    m.time = time
    m.options.SOLVER = 1 if integer else 3

    x = m.Var(value=x_0, name='x')
    i = m.Var(value=0.0, integer=integer, lb=0, ub=1, name='i') 

    m.Equation(x.dt() == f(x, i))
    m.Obj(F(x, i))

    m.options.IMODE = 6
    m.solve(disp=False)
    
    history = {}
    history['time'] = time
    history['x'] = x
    history['i'] = i
    history['obj_val'] = m.options.OBJFCNVAL
    history['x_0'] = x_0
    history['x_ref'] = x_ref
    return history

def optimal_control_substituting(x_0, x_ref, time, f, F, const_i):
    """
    Substitute the particular controls i(t) into the problem and 
    compute the objective value F
    .---------------------------------------.
    |   F - ?  s.t.  x(0) = x_0             |
    |                x(t+1) = f(x(t), i(t)) |
    .---------------------------------------.   
    params:
        x_0 - initial state
        x_ref - final reference state  
        time - time horizon
        f - differential equation
        F - objective function
        const_i - fixed sequence of i(t)
    returns:
        history: a dict with
            x - predicted state trjectory of x(t)
            i - given sequense of controls i(t)
            obj_val - corresponding objective value F(x, i)
            x_0, x_ref, time
    """
    m = GEKKO(remote=False)
    m.time = time

    x = m.Var(value=x_0, name='x')
    i = m.Param(const_i, name='i') 
    obj_val = m.Var(value=0.0)
    
    m.Equation(x.dt() == f(x, i))
    m.Equation(obj_val == F(x, i))

    m.options.IMODE = 4
    m.solve(disp=False)

    history = {}
    history['time'] = time
    history['x'] = x
    history['i'] = i
    history['obj_val'] = sum(obj_val.VALUE)
    history['x_0'] = x_0
    history['x_ref'] = x_ref
    return history
