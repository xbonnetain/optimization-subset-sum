
"""
Optimization target: new quantum walk algorithm with a 5-level merging tree
and {0,-1,1,2} representations (Section 6). This is the non heuristic case.

Run:
>>> optimize_quantum()

To obtain the exponent 0.218 and the parameters given in Section 6.3

"""

from macros import round_to_str, check_constraints, round_upwards_to_str, wrap
from macros import f,g,p_good_2_down, p_good_2_up, p_good
import collections
from math import*
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np


set_quantum = collections.namedtuple('quantum', 'p0 p1 p2 l0 l1 l2 l3 l4 c1 c2 c3 alpha1 alpha2 alpha3 gamma1')
def quantum(f) : return wrap(f,set_quantum)
#=================================
# Among these variables:
# - pi is the filtering probability at level i
# - li is the list size at level i
# - l0 < 0, since only a few nodes contain a solution
# - there is no asymmetric left-right split in this variant
# - ci is the total number of bits of the modular constraint at level i
# - alphai is the total number of "-1" at level i
# - gammai is the total number of "2" at level i


constraints_quantum = [
# filtering terms (they are negative variables)
{ 'type' : 'eq', 'fun' : quantum(lambda x : p_good_2_down(1/2., 0, 0., 1/4.+x.alpha1-2*x.gamma1, x.alpha1, x.gamma1) - x.p0 )},
{ 'type' : 'eq', 'fun' : quantum(lambda x : p_good_2_up(1/4.+x.alpha1-2*x.gamma1, x.alpha1, x.gamma1, 1/8.+x.alpha2, x.alpha2, 0) - x.p1)},
{ 'type' : 'eq', 'fun' : quantum(lambda x : p_good(1/8.+x.alpha2, x.alpha2, 1/16.+x.alpha3, x.alpha3) - x.p2)},
# sizes of the lists
{ 'type' : 'eq', 'fun' : quantum(lambda x : 2*x.l1 - (1-x.c1) + x.p0 - x.l0 )},
{ 'type' : 'eq', 'fun' : quantum(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1 )},
{ 'type' : 'eq', 'fun' : quantum(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2 )},
{ 'type' : 'eq', 'fun' : quantum(lambda x : 2*x.l4 - (x.c3 ) - x.l3 )},
# at other levels
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l0/2 + f(1/4.+x.alpha1-2*x.gamma1, x.alpha1, x.gamma1) - x.c1 - x.l1)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l0/4 + g(1/8.+x.alpha2, x.alpha2) - x.c2 - x.l2)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l0/8 + g(1/16.+x.alpha3, x.alpha3) - x.c3 - x.l3)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l0/16 + g(1/16.+x.alpha3, x.alpha3)/2 - x.l4)},
# coherence of the -1
#{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.alpha2 - x.alpha1/2)},
#{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.alpha3 - x.alpha2/2)},
#{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.alpha4 - x.alpha3/2)},
#{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.alpha1 - x.gamma1)},
# no update explosion
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.c3 - x.l4)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l4-x.l3)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l3-x.l2)},
{ 'type' : 'ineq', 'fun' : quantum(lambda x : x.l2-x.l1)}
]


def quantum_time_without_heuristic(x):
    """
    Quantum time without any heuristic on the quantum walk (it modifies the
    update cost). Now only the classical subset-sum heuristic is needed.
    """
    x = set_quantum(*x)
    setup = max( x.c3, x.l3, x.l2 - x.p2/2, x.l1 - x.p1/2, (x.l1 + max(x.l1 - (1-x.c1), 0))/2 )

    time2 = max(x.l3 - (x.c2-x.c3),0)
    # number of elements to modify at level 2
    elts2 = max(x.l3 - (x.c2-x.c3) + x.p2/2, 0)

    time1 = max(x.l2 - (x.c1-x.c2),0)
    # number of elements to modify at level 1
    elts1 = max(x.l2 - (x.c1-x.c2) + x.p1/2, 0)

    update = max(0, time2, elts2+time1, (elts2+elts1)/2 + max((x.l1 - (1 - x.c1))/2,0))
    return max(setup, (max(0, -x.l0) + x.l4)/2 + update)



def optimize_quantum_without_heuristic(verb=True):
    """
    Optimizes the parameters.
    """

    time = quantum_time_without_heuristic
    objective = time

    start = [-0.2, -0.03, -0.016,-0.2] + [0.18,0.21,0.22,0.22,0.63,0.43,0.22] + [(0.05)]*2 + [(0.005)]*2
    bounds = [(-1,0)]*4 + [(0,1)]*7 + [(0, 0.1)]*3 + [(0, 0.01)]

    result = opt.minimize(objective, start, 
            bounds= bounds, tol=1e-10, 
            constraints=constraints_quantum, options={'maxiter':10000})
    
    astuple = set_quantum(*result.x)
    
    if verb:
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        for t in astuple._asdict():
            print(t, round_to_str(astuple._asdict()[t]) )
        print(check_constraints(constraints_quantum, result.x))

    return result.success, objective(astuple), result


if __name__ == "__main__":
    print("=========== QUANTUM OPTIMIZATION WITHOUT QW HEURISTIC ===========")
    optimize_quantum_without_heuristic()

    pass

