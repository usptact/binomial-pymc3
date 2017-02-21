#
# Demo to attempt recover parameter p of the Binomial distribution.
#

import sys

import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

from scipy.stats import binom

p_true = 0.37
n = 10000
K = 50

X = binom.rvs( n=n, p=p_true, size=K )
print( X )

model = pm.Model()

with model:
    p = pm.Beta( 'p', alpha=2, beta=2 )
    y_obs = pm.Binomial( 'y_obs', p=p, n=n, observed=X )
    step = pm.Metropolis()
    trace = pm.sample( 10000, step=step, progressbar=True )

pm.traceplot( trace )
plt.show()
