# Sharpe Ratio and other Portfolio Statistics

# Daily Portfolio Values
# Starting Value, Allocations, Symbols
# Take prices, normalize them according to first price.
# Prices,
# Normed,
# Normed * alloced,
# alloced * start_val = position_values,
# Summing all position values for a day,
# gives us the total value of the portfolio
# pos_vals.sum(axis=1) = porfolio_value per day

# Portfolio Statistics
# daily_returns - first value will always be 0
# daily_rets = daily_rets[1:]
# cumulative return = port_val[-1]/port_val[0] -1
# avg daily return = daily_rets.mean()
# std deviation daily return = daily_rets.std()
# sharp ratio
# Return, Volatility - order of importance

# Sharpe Ratio
# Risk adjusted return
# considers risk free rate of return
# like bank account or short term treasury
# Portfolio return, Rp
# risk free rate of return, Rf
# std dev of portfolio return stdo
# Portfolio return - risk free rate divided by std dev
# std dev = volatility

# Computing Sharp Ratio
# Expected Value[Rp - Rf]/std[Rp-Rf]
# mean(daily_rets - daily_riskfree)/std(daily_rets - daily_riskfree)
# daily_riskfree -> LIBOR, 3mo treasury bill, or 0%
# 252_root of 1 + risk rate to get daily rate from annual.
# mean(daily_rets - daily_riskfree)/std(daily_rets) if risk free rate is constant

# Sharpe ratio can vary widely depending on how frequently you sample
# monthly daily yearly

# SRannualized = K * SR
# K = square root of samples per year (252 for daily, 52 for weekly),
# we still use 252 if we only traded for 89 days

# Important - 
# Cumulative Return
# Average Daily Return
# Risk
# Sharpe Ratio
# Make Functions for these
