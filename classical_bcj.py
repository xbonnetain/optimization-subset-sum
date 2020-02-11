
"""
Optimization target: BCJ algorithm with relaxed constraints
(Section 2.4, second paragraph).

Run:
>>> optimize_bcj_classical()

To obtain the exponent 0.289

"""


from macros import*
import collections
from math import*
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np


set_bcj = collections.namedtuple('BCJ', 'p0 p1 p2 l1 l2 l3 l4 c1 c2 c3 alpha1 alpha2 alpha3')
def bcj(f) : return wrap(f,set_bcj)
#=================================
# Among these variables:
# - pi is the filtering probability at level i
# - li is the list size at level i (classically, all lists at a given level
# have same size)
# (l0 = 0, since we want to find the solution)
# - ci is the total number of bits of the modular constraint at level i
# - alphai is the total number of "-1" at level i


constraints_bcj_classical = [
# filtering terms
{ 'type' : 'eq', 'fun' : bcj(lambda x : p_good(1/2., 0., 1/4.+x.alpha1, x.alpha1) - x.p0 )},
{ 'type' : 'eq', 'fun' : bcj(lambda x : p_good(1/4.+x.alpha1, x.alpha1, 1/8.+x.alpha2, x.alpha2) - x.p1)},
{ 'type' : 'eq', 'fun' : bcj(lambda x : p_good(1/8.+x.alpha2, x.alpha2, 1/16.+x.alpha3, x.alpha3) - x.p2)},
# sizes of the lists
{ 'type' : 'eq', 'fun' : bcj(lambda x : 2*x.l1 - (1-x.c1) + x.p0 )},
{ 'type' : 'eq', 'fun' : bcj(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1 )},
{ 'type' : 'eq', 'fun' : bcj(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2 )},
{ 'type' : 'eq', 'fun' : bcj(lambda x : 2*x.l4 - x.c3 - x.l3 )},
{ 'type' : 'ineq', 'fun' : bcj(lambda x : g(1/4.+x.alpha1, x.alpha1) - x.c1 - x.l1)},
{ 'type' : 'ineq', 'fun' : bcj(lambda x : g(1/8.+x.alpha2, x.alpha2) - x.c2 - x.l2)},
{ 'type' : 'ineq', 'fun' : bcj(lambda x : g(1/16.+x.alpha3, x.alpha3) - x.c3 - x.l3)},
{ 'type' : 'ineq', 'fun' : bcj(lambda x : g(1/16.+x.alpha3, x.alpha3)*0.5 - x.l4)},
# coherence of the -1
{ 'type' : 'ineq', 'fun' : bcj(lambda x : x.alpha2 - x.alpha1/2)},
{ 'type' : 'ineq', 'fun' : bcj(lambda x : x.alpha3 - x.alpha2/2)}
]


def classical_time_bcj(x):
    x = set_bcj(*x)
    return max(x.l4, x.l3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)


def optimize_bcj_classical(verb=True):
    """
    Optimizes the classical BCJ algorithm.
    """

    time = classical_time_bcj
    objective = time
    mycons = constraints_bcj_classical
    
    start = [(-0.2)]*3 + [(0.2)]*10
    bounds = [(-1,0)]*3 + [(0,1)]*10
    
    result = opt.minimize(time, start, 
            bounds= bounds, tol=1e-10, 
            constraints=mycons, options={'maxiter':5000})
    
    astuple = set_bcj(*result.x)
    
    if verb:
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        
        for t in astuple._asdict():
            print(t, round_to_str(astuple._asdict()[t]) )
        print("Checking that the constraints are satisfied:")
        print(check_constraints(mycons, result.x))

    return result.success, objective(astuple), result

if __name__ == "__main__":

    print("=========== CLASSICAL OPTIMIZATION ===========")
    optimize_bcj_classical()

    pass


