"""Test module"""

from numpy import np
from authentic_performance import kendall_tauDistance

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

p1 = np.arange(100)
assert kendall_tauDistance(p1,p1) == 0

p1 = np.arange(100)
p2 = np.arange(100)
p2[0] = 1
p2[1] = 0
assert kendall_tauDistance(p1,p2) == 1

p1 = np.arange(100)
p2 = np.arange(100)
p2[0] = 1
p2[1] = 0
p2[2] = 3
p2[3] = 2
assert kendall_tauDistance(p1,p2) == 2

p1 = np.arange(100)
p2 = np.arange(100)
p2[0] = 2
p2[1] = 1
p2[2] = 0
assert kendall_tauDistance(p1,p2) == 3
