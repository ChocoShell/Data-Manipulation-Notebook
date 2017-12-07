# What is portfolio optimization?
"""
Given a set of assets and a time period,
    find an allocation of funds to assets that
    maximizes performance.

What is performance?
    cum ret
    volatility
    risk adjusted return
        - sharpe ratio
"""

# The difference optimization can make
"""
Equal parts in GOOG, AAPL, GLD, XOM
    compare to the SPY
Optimized 0 Goog, 40 AAPL, 60 GLD, 0 XOM
"""

# Which criteria is easiest to sovle for?
"""
cum return
min volatility
sharpe ratio

Cumulative Return is easiest
"""

# Framing the Problem
"""
- Provide a function to minimize f(x)
- provide an inital guess for x
- call the optimizer

x is the allocations
We want largest sharpe ratio or smallest negative sharpe ratio.
"""

# Ranges and constraints
"""
- Ranges: Limits on values for x 0 <-> 1
- Constraints: Properties of the values of x that must be true.
    x0 + x1 + x2 + x3 = 1
"""
