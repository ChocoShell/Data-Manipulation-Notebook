## Incomplete Data

# Pristine Data
"""
 Data comes from many sources
 It's not perfect minute by minute
 It has tons of gaps and missing data points
 Different stock prices at the same time from different exchanges
 Not all stocks trade everyday
"""

# Why does data go missing
# SPY is the reference.  If SPY trades, the stock market is open. 1993
# Stocks can changes names/tickers

# What do we do when we are missing data?
# We fill forward on days the stock doesn't trade
# At the beginning of the stock, fill backward.

# Pandas fillna()
# Fill forward
df.fillna(method='ffill')
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.DataFrameGroupBy.fillna.html?highlight=fillna#pandas.core.groupby.DataFrameGroupBy.fillna

df_data.fillna(method='ffill', inplace=True)

df_data.fillna(method='bfill', inplace=True)
