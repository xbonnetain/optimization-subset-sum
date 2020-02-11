
"""
Optimization target: new algorithm for subset-sum, using {0,-1,1,2} representations
with a 5-level merging tree.
(Section 2.4).

Run:
>>> optimize_classical()

To obtain the exponent 0.283

"""


from macros import round_to_str, check_constraints, round_upwards_to_str, wrap
import collections
from math import*
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np


set_classical = collections.namedtuple('classical', 'p0 p1 p2 l1 l2 l3 l4 c1 c2 c3 alpha1 alpha2 alpha3 gamma1 gamma2 gamma3')
def classical(f) : return wrap(f,set_classical)
#=================================
# Among these variables:
# - pi is the filtering probability at level i
# - li is the list size at level i (classically, all lists at a given level
# have same size)
# (l0 = 0, since we want to find the solution)
# - ci is the total number of bits of the modular constraint at level i
# - alphai is the total number of "-1" at level i
# - gammai is the total number of "2" at level i


def xlx(x):
    if x<=0: return 0
    return x*log(x, 2)


def p_good(a0, b0, a1, b1):
    return -2*xlx(a0/2) - 2*xlx(b0/2) - xlx(a1-a0/2) - xlx(b1-b0/2) - xlx(1-a1-b1-a0/2-b0/2) - 2*g(a1, b1)
    

def g(a, b):
    return -xlx(a) - xlx(b) - xlx(1-a-b)


def f(a, b, c):
    if a<=0: return g(b, c)
    if b<=0: return g(a, c)
    if c<=0: return g(a, b)
    if a+b+c >= 1: return min(g(b, c), g(a, c), g(a, b))
    try:
        return -a*log(a,2) - b*log(b,2) - c*log(c,2) - (1-a-b-c)*log(1-a-b-c, 2)
    except:
        return 0.

    
def p_good_2(b0, a0, c0, b1, a1, c1):
    def proba(x):
        return 2*xlx(a0/2) + 2*xlx(x+a1-a0/2-b0/2) + xlx(1-c0-2*a1-2*x) + 2*xlx(b0/2-x) + 2*xlx(x) + 2*xlx(x+c0/2-b1/2+a1/2-a0/4-b0/4) + xlx(b1-a1+a0/2+b0/2-2*x)
    bounds = [( max(a0/2+b0/2-a1, 0, b1/2-a1/2+a0/4+b0/4-c0/2), min(1/2.-c0/2-a1, b0/2, b1/2-a1/2+a0/4+b0/4) )]
    if bounds[0][0] > bounds[0][1]: return p_good(b0, a0, b1, a1) - 1
    return - opt.fminbound(proba, bounds[0][0], bounds[0][1], xtol=1e-15, full_output=1)[1] - 2*f(a1, b1, c1)


def p_good_2_aux(b0, a0, c0, b1, a1, c1):
    return -( 2*xlx(a1-c1) + xlx(1-2*c1-2*b1) + 2*xlx(c1) + 2*xlx(b0/2-c1) ) - 2*f(a1, b1, c1)


constraints_classical = [
# filtering terms
{ 'type' : 'eq', 'fun' : classical(lambda x : p_good_2_aux(1/2., 0, 0., 1/4.+x.alpha1-2*x.gamma1, x.alpha1, x.gamma1) - x.p0 )},
{ 'type' : 'eq', 'fun' : classical(lambda x : p_good_2(1/4.+x.alpha1-2*x.gamma1, x.alpha1, x.gamma1, 1/8.+x.alpha2-2*x.gamma2, x.alpha2, x.gamma2) - x.p1)},
{ 'type' : 'eq', 'fun' : classical(lambda x : p_good_2(1/8.+x.alpha2-2*x.gamma2, x.alpha2, x.gamma2, 1/16.+x.alpha3-2*x.gamma3, x.alpha3, x.gamma3) - x.p2)},
# sizes of the lists
{ 'type' : 'eq', 'fun' : classical(lambda x : 2*x.l1 - (1-x.c1) + x.p0 )},
{ 'type' : 'eq', 'fun' : classical(lambda x : 2*x.l2 - (x.c1 - x.c2) + x.p1 - x.l1 )},
{ 'type' : 'eq', 'fun' : classical(lambda x : 2*x.l3 - (x.c2 - x.c3) + x.p2 - x.l2 )},
{ 'type' : 'eq', 'fun' : classical(lambda x : 2*x.l4 - x.c3 - x.l3 )},
{ 'type' : 'ineq', 'fun' : classical(lambda x : f(1/4.+x.alpha1-2*x.gamma1, x.alpha1, x.gamma1) - x.c1 - x.l1)},
{ 'type' : 'ineq', 'fun' : classical(lambda x : f(1/8.+x.alpha2-2*x.gamma2, x.alpha2, x.gamma2) - x.c2 - x.l2)},
{ 'type' : 'ineq', 'fun' : classical(lambda x : f(1/16.+x.alpha3-2*x.gamma3, x.alpha3, x.gamma3) - x.c3 - x.l3)},
{ 'type' : 'ineq', 'fun' : classical(lambda x : f(1/16.+x.alpha3-2*x.gamma3, x.alpha3, x.gamma3)*0.5 - x.l4)},
# coherence of the -1
{ 'type' : 'ineq', 'fun' : classical(lambda x : x.alpha2 - x.alpha1/2)},
{ 'type' : 'ineq', 'fun' : classical(lambda x : x.alpha3 - x.alpha2/2)},
{ 'type' : 'ineq', 'fun' : classical(lambda x : x.alpha1 - 2*x.gamma1)},
{ 'type' : 'ineq', 'fun' : classical(lambda x : x.alpha2 - 2*x.gamma2)},
{ 'type' : 'ineq', 'fun' : classical(lambda x : x.alpha3 - 2*x.gamma3)},
# if no 2
#{ 'type' : 'ineq', 'fun' : classical(lambda x : - x.gamma1)},
#{ 'type' : 'ineq', 'fun' : classical(lambda x : - x.gamma2)},
#{ 'type' : 'ineq', 'fun' : classical(lambda x : - x.gamma3)},
]


def classical_time(x):
    x = set_classical(*x)
    return max(x.l4, x.l3, x.l2 - x.p2, x.l1 - x.p1, -x.p0)


def optimize_classical(verb=True):
    """
    Optimizes the classical algorithm.
    """

    time = classical_time
    objective = time
    mycons = constraints_classical
    
    start = [(-0.2)]*3 + [0.2]*7 + [0.03]*3 + [0.005]*3
    bounds = [(-1,0)]*3 + [(0,1)]*7 + [(0, 0.05)]*3 + [(0, 0.01)]*3
    
    result = opt.minimize(time, start, 
            bounds= bounds, tol=1e-10, 
            constraints=mycons, options={'maxiter':5000})
    
    astuple = set_classical(*result.x)
    
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
    optimize_classical()

    pass

