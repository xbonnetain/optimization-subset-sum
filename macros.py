"""
This file defines some functions used in our optimizations.
"""

from math import*
import scipy.optimize as opt


def check_constraints(constraints, solution) : 
    return [ (constraint['type'], constraint['fun'](solution)) for constraint in constraints ]


def wrap(f,g) :
    def inner(x):
        return f(g(*x))
    return inner


#=================================
# macros used in quantum_qw.py and quantum_qw_no_heuristic.py
#====================================

def xlx(x):
    if x<=0: return - 100*x
    return x*log(x, 2)


#def p_good(a0, b0, a1, b1):
#    """
#    Probability (in log_2 and proportion of n) that a pair of subknapsacks
#    having a1 "1" and b1 "-1" sum to a valid subknapsack having a0 "1" and b0 "-1".
#    """
#    return -2*xlx(a0/2) - 2*xlx(b0/2) - xlx(a1-a0/2) - xlx(b1-b0/2) - xlx(1-a1-b1-a0/2-b0/2) - 2*g(a1, b1)




def f(a, b, c):
    return -xlx(a) - xlx(b) - xlx(c) - xlx(1-a-b-c)


def g(a, b):
    return -xlx(a) - xlx(b) - xlx(1-a-b)


def p_good_2_down(b0, a0, c0, b1, a1, c1):
    return -( 2*xlx(a1-c1) + xlx(1-2*c1-2*b1) + 2*xlx(c1) + 2*xlx(b0/2-c1) ) - 2*f(a1, b1, c1)


def p_good_2_up(b0, a0, c0, b1, a1, c1):
    return -( 2*xlx(a0/2) + 2*xlx(a1-a0/2) + xlx(1-b0-c0-2*a1) + 2*xlx(b0/2) + xlx(c0) ) - 2*f(b1, a1, c1)


def p_good(a0, b0, a1, b1):
    """
    Probability (in log_2 and proportion of n) that a pair of subknapsacks
    having a1 "1" and b1 "-1" sum to a valid subknapsack having a0 "1" and b0 "-1".
    """
    return -2*xlx(a0/2) - 2*xlx(b0/2) - xlx(a1-a0/2) - xlx(b1-b0/2) - xlx(1-a1-b1-a0/2-b0/2) - 2*g(a1, b1)
    

def round_to_str(t):
    """
    Rounds the value 't' to a string with 4 digit precision (adding trailing zeroes
    to emphasize precision).
    """
    s = str(round(t,4))
    # must be 6 digits
    return (s + "0" * (5 + s.find(".") -len(s)))
        

def round_upwards_to_str(t):
    """
    Rounds the value 't' *upwards* to a string with 4 digit precision (adding trailing zeroes
    to emphasize precision).
    """
    s = str( ceil(t*10000)/10000 )
    # must be 6 digits
    return (s + "0" * (5 + s.find(".") -len(s)))


