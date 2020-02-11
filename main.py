
"""
Runs separately all our optimization programs and shows our results.
"""

from macros import *
from classical_bcj import optimize_bcj_classical
from classical import optimize_classical
from quantum_qw import optimize_quantum
from quantum_qw_no_heuristic import optimize_quantum_without_heuristic
from quantum_hgj_asymmetric import optimize_hgj


print("======= RUNNING ALL OPTIMIZATIONS ============")

results = [
(optimize_bcj_classical(verb=False)[1], "Classical BCJ algorithm with relaxed constraints (Section 2.4, second paragraph)"),
(optimize_classical(verb=False)[1], "New classical algorithm with {-1,0,1,2} representations (Section 2.4)."),
(optimize_hgj(verb=False,flag="classical")[1], "Classical HGJ algorithm"),
(optimize_hgj(verb=False,flag="quantum1")[1], "Quantum asymmetric HGJ algorithm (Section 4.2)"),
(optimize_hgj(verb=False,flag="quantum2")[1], "Quantum asymmetric HGJ algorithm with quantum filtering (Section 4.3)"),
(optimize_quantum(verb=False)[1], "New heuristic quantum walk algorithm (Section 5.3)"),
(optimize_quantum_without_heuristic(verb=False)[1], "New non-heuristic quantum walk algorithm (Section 6.3)")
]

print("======= SHOWING ALL OPTIMIZATION RESULTS ============")


for t in results:
    print(t[1]," : ", round_upwards_to_str(t[0]))

