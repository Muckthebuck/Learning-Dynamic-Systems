"""Module for keeping benchmarking and test systems in one place so that they can be reused."""
from indirect_identification.armax import ARMAX

# True parameters
a1_true = 0.7
a2_true = 0.2
b1_true = 0.4
b2_true = 0.3

# First order open-loop system
A = [1, a1_true]
B = [b1_true]
C = [1]
F = [0]
L = [1]

OpenLoop1 = ARMAX(A, B, C, F, L)

# Second order open-loop system
A = [1, a1_true, a2_true]
B = [b1_true, b2_true]
C = [1]
F = [0]
L = [1]

OpenLoop2 = ARMAX(A, B, C, F, L)

# First order closed-loop system
A = [1, a1_true]
B = [b1_true]
C = [1]
F = [1]
L = [1]

ClosedLoop1 = ARMAX(A, B, C, F, L)


# Second order closed-loop system
A = [1, a1_true, a2_true]
B = [b1_true, b2_true]
C = [1]
F = [1]
L = [1]

ClosedLoop2 = ARMAX(A, B, C, F, L)
