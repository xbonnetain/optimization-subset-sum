
"""
Optimization target: new quantum walk algorithm with a 6-level merging tree
and {0,-1,1,2} representations (Section 5). This is the heuristic case.

Run:
>>> optimize_quantum()

To obtain the exponent 0.216 and the parameters given in Section 5.3

"""

from macros import round_to_str, check_constraints, round_upwards_to_str, wrap
from macros import f,g,p_good_2_down, p_good_2_up, p_good
import collections
from math import*
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np


set_quantum = collections.namedtuple('quantum', 'p0 p1 p2 p3 l0 l1 l2 l3 l4 delta c1 c2 c3 c4 alpha1 alpha2 alpha3 alpha4 gamma1')
def quantum(f) : return wrap(f,set_quantum)
#=================================
# Among these variables:
# - pi is the filtering probability at level i
# - li is the list size at level i
# - l0 < 0, since only a few nodes contain a solution
# - gamma is the parameter of the asymmetric left-right split between l5r and l5l
# - l5r and l5l are implicit. We have l5l = l4 in size and l5r = c4
# - ci is the total number of bits of the modular constraint at level i
# - alphai is the total number of "-1" at level i
# - gammai is the total number of "2" at level i


constraints_quantum = [
# filtering terms (they are negative variables)
{ 'type' : 'eq', 'fun' : quantum(lambda x : p_good_2_down(1/2., 0, 0., 1/4.+x.alpha1-2*x.gamma1, x.alpha1, x.gamma1) - x.p0 )},
{ 'type' : 'eq', 'fun' : quantum(lambda x : p_good_2_up(1/4.+x.alpha1-2*x.gamma1, x.alpha1, x.gamma1, 1/8.+x.alpha2, x.alpha2, 0) - x.p1)},
{ 'type' : 'eq', 'fun' : quantum(lambda x : p_good(1/8.+x.alpha2, x.alpha2, 1/16.+x.alpha3, x.alpha3) - x.p2)},
{ 'type' : 'eq', 'fun' : quantum(lambda x : p_good(1/16.+x.alpha3, x.alpha3, 1/32.+x.alpha4, x.alpha4) - x.p3)},
# sizes of the lists
{ 'type' : 'eq', 'fun' : quantum(lambda x : 2*x.l1 - (1-x.c1) + x.p0 - x.l0 )},
{ 'type' : 'eq', 'fun' : quantum(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1 )},
{ 'type' : 'eq', 'fun' : quantum(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2 )},
{ 'type' : 'eq', 'fun' : quantum(lambda x : 2*x.l4 - (x.c3 - x.c4) + x.p3 - x.l3 )},
# size of l5r = c4
{ 'type' : 'eq', 'fun' : quantum(lambda x : g(1/32.+x.alpha4, x.alpha4)*(1-x.delta) - x.c4 )},
# size of l5l = l4
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l0/16 + g(1/32.+x.alpha4, x.alpha4)*x.delta - x.l4)},
# at other levels
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l0/2 + f(1/4.+x.alpha1-2*x.gamma1, x.alpha1, x.gamma1) - x.c1 - x.l1)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l0/4 + g(1/8.+x.alpha2, x.alpha2) - x.c2 - x.l2)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l0/8 + g(1/16.+x.alpha3, x.alpha3) - x.c3 - x.l3)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l0/16 + g(1/32.+x.alpha4, x.alpha4) - x.c4 - x.l4)},
# coherence of the -1
#{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.alpha2 - x.alpha1/2)},
#{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.alpha3 - x.alpha2/2)},
#{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.alpha4 - x.alpha3/2)},
#{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.alpha1 - x.gamma1)},
# no update explosion
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l4-x.l3)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l3-x.l2)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l2-x.l1)},
]


def quantum_time(x):
    """
    Heuristic quantum time.
    """
    x = set_quantum(*x)
    setup = max( x.c4, x.l4, x.l3 - x.p3/2, x.l2 - x.p2/2, x.l1 - x.p1/2, (x.l1 + max(x.l1 - (1-x.c1), 0))/2 )
    update = max( 0, (x.l4 - (x.c3 - x.c4))/2, (x.l3 - (x.c2 - x.c3))/2, (x.l2 - (x.c1 - x.c2))/2, (x.l1 - (1 - x.c1))/2)
    return max(setup, (max(0, -x.l0) + x.l4)/2 + update)


def optimize_quantum(verb=True):
    """
    """
    
    time = quantum_time
    
    objective = time
    mycons = constraints_quantum

    start = [-0.2, -0.03, -0.02, 0, -0.2] + [(0.19)]*9 + [(0.05)]*3 + [(0.005)]*2
    bounds = [(-1,0)]*5 + [(0,1)]*9 + [(0, 0.1)]*4 + [(0, 0.01)]
   
    result = opt.minimize(objective, start, 
            bounds= bounds, tol=1e-10, 
            constraints=mycons, options={'maxiter':5000})
    
    astuple = set_quantum(*result.x)
    
    if verb:
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_to_str(astuple._asdict()[t]) )
        print(check_constraints(mycons, result.x))

    return result.success, objective(astuple), result


if __name__ == "__main__":
    print("=========== QUANTUM OPTIMIZATION WITH QW HEURISTIC ===========")
    optimize_quantum()

    pass

