
"""
Optimization target: new quantum subset-sum algorithm based on an "asymmetric"
HGJ merging tree (Section 4 of the paper) and its trade-offs.

Use:
>>> optimize_hgj("classical")
to obtain the exponent 0.337 of the classical HGJ algorithm

>>> optimize_hgj("quantum1")
to obtain the exponent 0.2374 of our first quantum version (Section 4.2)

>>> optimize_hgj("quantum2")
to obtain the exponent 0.2356 of our second quantum version (Section 4.3),
using quantum filtering

>>> optimize_hgj("moremem")
to optimize when more classical RAM is allowed than the memory bound.

>>> show_optimizations()
to show all optimizations

>>> print_table_contents()
to print the contents of table 2 (time-memory tradeoff with QRACM constraint)
in Section 4.4 of the paper

>>> create_graph()
to create tikz code for the plot of Figure 4 in Section 4.4

"""

from macros import round_to_str, check_constraints, round_upwards_to_str, wrap
import collections
from math import*
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np


# in these parameters, c0 is the size (in bits, in proportion of n) of the final
# modular constraint that we obtain; i.e., if we want a solution of the
# knapsack problem, we should enforce c0 = 1
set_qhgj = collections.namedtuple('QHGJ', 'l30 l31 l32 l34 l20 l21 l22 l10 l11 a b c r c20 c21 c1 c0')
def qhgj(f) : return wrap(f,set_qhgj)


def xlx(x):
    if x<=0: return 0
    return x*log(x, 2)


def h(a):
    """
    Hamming entropy.
    """
    return -xlx(a) - xlx(1-a)


def filtering(a,b):
    """
    Returns (in log_2, and in proportion of n) the probability that two
    binary vectors such that:
    - the first one is uniformly sampled at random from all vectors having
    Hamming weight "an"
    - the second one is uniformly sampled at random from all vectors having
    Hamming weight "bn"
    have two colliding "ones". This is the filtering of representations in 
    the HGJ algorithm and its variants.
    """
    return (1-b)*h(a/(1-b)) - h(a)


constraints_hgj = [
# weight of the solution
{ 'type' : 'eq', 'fun' : qhgj(lambda x :  x.a*2 + x.b + x.c -0.5)},
# increasing constraints
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  x.c1 - x.c21)},
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  x.c1 - x.c20)},
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  x.c0 - x.c1)},
# saturation at level 0
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  h(x.c) * (1-x.r) -x.l30)},
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  h(x.c) * x.r -x.l31)},
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  h(x.b) / 2 -x.l32)},
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  h(x.a) / 2 -x.l34)},
# saturation at level 1
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  h(x.c) - x.c20 - x.l20)},
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  h(x.b) - x.c20 - x.l21)},
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  h(x.a) - x.c21 - x.l22)},
# saturation at level 2
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  h(x.c + x.b) - x.c1 - x.l10)},
{ 'type' : 'ineq', 'fun' : qhgj(lambda x :  h(2*x.a) - x.c1 - x.l11)},
# merging at level 1: no filtering
{ 'type' : 'eq', 'fun' : qhgj(lambda x :  x.l30 + x.l31 - x.c20 - x.l20)},
{ 'type' : 'eq', 'fun' : qhgj(lambda x :  x.l32*2 - x.c20 - x.l21)},
{ 'type' : 'eq', 'fun' : qhgj(lambda x :  x.l34*2 - x.c21 - x.l22)},
# merging at level 2, with filtering
{ 'type' : 'eq', 'fun' : qhgj(lambda x :  x.l20 + x.l21 - x.c1 + x.c20 + filtering(x.b, x.c) - x.l10)},
{ 'type' : 'eq', 'fun' : qhgj(lambda x :  x.l22*2 - x.c1 + x.c21 + filtering(x.a, x.a) - x.l11)},
# merging at level 3, with filtering
{ 'type' : 'eq', 'fun' : qhgj(lambda x :  x.l10 + x.l11 - (x.c0-x.c1) + filtering(x.b + x.c, 2*x.a))}
]


def memory_hgj(x):
    x = set_qhgj(*x)
    return max(x.l31,x.l32,x.l34,x.l21,x.l22,x.l11)


def classical_time_hgj(x):
    x = set_qhgj(*x)
    return (1 - x.c0) + max(x.l31, x.l32, x.l34, x.l21, x.l22, x.l22*2 - x.c1 + x.c21,
                x.l30 + max(x.l31 - x.c20,0) + max(x.l21 - x.c1 + x.c20,0 ))


def quantum_time_hgj_first(x):
    """
    Quantum time in the QRACM model, as obtained in section 4.2
    """
    x = set_qhgj(*x)
    return (1 - x.c0) + max(x.l31, x.l32, x.l34, x.l21, x.l22, x.l22*2 - x.c1 + x.c21,
            0.5*(  x.l10 - filtering(x.b, x.c) + max(x.c20 - x.l31, 0) + max(x.c1 - x.c20 - x.l21, 0 ) ))


def quantum_time_hgj_second(x):
    """
    Improved quantum time in the QRACM model, as obtained in section 4.3, 
    using quantum filtering in the intermediate lists.
    """
    x = set_qhgj(*x)
    return (1 - x.c0) + max(x.l31, x.l32, x.l34, x.l21, x.l22, x.l22*2 - x.c1 + x.c21 + filtering(x.a, x.a)*0.5,
            0.5*(  x.l10 - filtering(x.b, x.c) + max(x.c20 - x.l31, 0) + max(x.c1 - x.c20 - x.l21, 0 ) ))


def optimize_hgj(flag="classical", verb=True, mcons=None):
    """
    Optimizes the parameters of our "asymmetric" quantum HGJ algorithm and
    returns the best time complexity achievable.
    
    @param flag: either "classical", "quantum1", "quantum2", "moremem" where "classical"
    means the classical optimization (we should find the original parameters),
    "quantum1" means the first quantum optimization (without improved filtering),
    "quantum2" means the second (with improved filtering).
    "moremem" means that we are in the "quantum1" setting and the memory constraint
    is enforced only on the quantum-accessed lists. So the classical memory used
    may (and will) be higher than the memory constraint that we enforce.
    
    @param verb: decides if the results must be printed.
    @param mcons: if not none, specifies a memory constraint. For ex: mcons = 0.1
    means that the memory used should be  of the order 2^{0.1 n}
    """
    if flag not in ["classical", "quantum1", "quantum2", "moremem"]:
        raise ValueError("Invalid flag: " + str(flag))
    
    time = (classical_time_hgj if flag == "classical" else
            quantum_time_hgj_second if flag == "quantum2" else
            quantum_time_hgj_first)
    
    mycons = constraints_hgj[:]
    mycons.append( { 'type' : 'eq', 'fun' : qhgj(lambda x :  x.c0 - 1)} )
    
    objective = time
    # try to minimize the memory
    if mcons is not None:
        if flag == "moremem":
            # constrain only the QRACM
            mycons.append( {'type' : 'ineq', 'fun' : qhgj(lambda x : mcons - max(x.l31,x.l21,x.l11)) } )
        else:
            mycons.append( {'type' : 'ineq', 'fun' : qhgj(lambda x : mcons - max(x.l31,x.l32,x.l21,x.l22,x.l34,x.l11,x.l22)) } )

    result = opt.minimize(objective, [0.]*17, bounds=[(0,1)]*17, 
                tol=1e-8, constraints=mycons,options={ "maxiter" : 10000})
    astuple = set_qhgj(*result.x)
    
    if verb:
        print("Validity: ", result.success)
        print("Time: ", round_upwards_to_str(time(astuple)))
        print("Memory: ", round_upwards_to_str(memory_hgj(astuple)))
        print("Sum: ", round_upwards_to_str( time(astuple) + memory_hgj(astuple))) 
        for t in astuple._asdict():
            print(t, round_to_str(astuple._asdict()[t]) )
        print(check_constraints(mycons, result.x))

    return result.success, objective(astuple), result


def show_optimizations():
    """
    Shows the results of the classical and QRACM optimizations.
    """
    print("============= CLASSICAL ===================")
    optimize_hgj(flag="classical")
    
    print("============= QUANTUM =====================")
    optimize_hgj(flag="quantum1")
    
    print("=========== QUANTUM IMPROVED =============")
    optimize_hgj(flag="quantum2")
    
    print("=========== AUTHORIZING MORE MEMORY ===========")
    optimize_hgj(flag="moremem")


def print_table_contents():
    """
    Prints the contents of the QRACM time-memory tradeoff table in Section 4
    in the paper.
    """
    
    for m in [0.05, 0.1, 0.15, 0.2, 0.3]:
        l = [round_to_str(m)]
        for flag in ["quantum1", "quantum2", "moremem"]:
            a,b,c = optimize_hgj(flag=flag,verb=False,mcons=m)
            astuple = set_qhgj(*c.x)
            time = (classical_time_hgj if flag == "classical" else
            quantum_time_hgj_second if flag == "quantum2" else
            quantum_time_hgj_first)
            l.append(round_upwards_to_str(time(astuple)))
            l.append(round_upwards_to_str(memory_hgj(astuple)))
        print(" & ".join([t for t in l]) + "\\\\")


def graph_to_tex(xdata, ydata, ylabels, ytitle="Y title", xtitle="X title", title="example graph", out=None):
    """
    Exports a graph to tikz code (used for figures in the paper).
    """
    nbplots = len(ydata)
    
    stringplots = list()
    for i in range(nbplots):
        stringplots.append(' '.join( 
            [ "(%s, %s)" % (xdata[i][j], ydata[i][j] ) for j in range(len(xdata[i])) ] ))
    
    to_tex_plots = ''.join( [ 
"""
\\addplot coordinates {
%s
};
\\addlegendentry{%s}""" % (stringplots[i], ylabels[i]) for i in range(nbplots) ] )
    
    fig = """
\\begin{figure}[!htb]
\\centering
\\begin{tikzpicture}
\\begin{axis}[
scale=0.8,
legend pos=outer north east,
xlabel={%s},
ylabel={%s},
ymin=0,
xmajorgrids,
ymajorgrids,
title={%s},
legend pos=outer north east,
cycle list name=my black white,
legend style={cells={align=left},name=legend}
]
%s
\\end{axis}
\\node[anchor=north west] at (legend.south west) {
\\begin{tabular}{c}
The complexities \\\\ are $\\bigOt{2^{\\alpha n}}$
\\end{tabular}
};
\\end{tikzpicture}
\\caption{%s}
\\label{fig:}
\\end{figure}
    """ % (xtitle, ytitle, title,to_tex_plots,title )
    
    if out is not None:
        with open(out, 'w') as f:
            f.write(fig)
    else:
        print(fig)
    return fig


def create_graph():
    """
    Prints the QRACM time-memory tradeoff graph in Section 4 of the paper.
    Plots the "quantum2" optimization and the T*M = 2^n/2 curve.
    """
    
    legend = ["Optimization", "$T M = 2^{n/2}$"]
    xdata = [ [],[] ]
    ydata = [ [],[] ]

    for m in [0,0.025,0.05,0.075,0.1,0.15,0.175,0.2,0.225, 0.2358, 0.3]:
        a,b,c = optimize_hgj(flag="quantum2",verb=False,mcons=m)
        astuple = set_qhgj(*c.x)
        ydata[0].append(quantum_time_hgj_second(astuple))
        xdata[0].append(m)
        xdata[1].append(m)
        ydata[1].append( 0.5 - m)# if 0.5-m > 0.2358 else 0.2358 )
    
    graph_to_tex(xdata, ydata, legend, ytitle="", xtitle="", title="Time-memory tradeoff", out=None)


if __name__ == "__main__":
    
    show_optimizations()
    #print_table_contents()
    #create_graph()
    pass
    


