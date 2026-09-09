"""Same four scores, ten times bigger. Only the softmax changes.

    python experiments/softmax_saturation.py
"""

import math


def softmax(row):
    exps = [math.exp(x - max(row)) for x in row]
    total = sum(exps)
    return [round(e / total, 3) for e in exps]


small = [1.8, 2.4, 0.9, 1.1]
big = [18.0, 24.0, 9.0, 11.0]

print("small", softmax(small))
print("big  ", softmax(big))
