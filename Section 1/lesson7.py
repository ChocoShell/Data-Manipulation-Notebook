# Histograms and Scatter Plots

"""
Histogram of daily returns will have values near 0%
 in bell curve fashion
"""

# Kurtosis tells us about the tails of a distribution
# Daily returns are Gaussian/Normal
# Daily Returns have fat tails compared to normal.
# Positive kurtosis means fatter tails than normal.
# Negative Kurtosis means skinny tails.

# Plotting Histograms
import pandas as pd
import matplotlib.pyplot as plt

from util import get_data, plot_data

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:]/df[:-1].values) - 1
    daily_returns.ix[0,:] = 0 # Set daily returns for row 0 to 0
    return daily_returns

def test_run():
    # Read Data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY']
    df = get_data(symbols, dates)
    plot_data(df)

    # Compute Daily Returns
    daily_return = compute_daily_returns(df)
    plot_data(daily_returns)

    # Plot a histogram
    daily_returns.hist() # Default number of bins is 10
    daily_returns.hist(bins=20) # 20 bins

# Computing Histogram Statistics
    mean = daily_returns['SPY'].mean()
    std = daily_returns['SPY'].std()

    plt.axvline(mean)
    plt.axvline(std)
    plt.axvline(-std)
    plt.show()

    print daily_returns['SPY'].kurtosis()

# Compare two histograms
# Plot two symbols on same histogram
# Higher STD = higher volatility. Broad ends

# Scatter Plots
# Form relationships between two symbols. Usually Linear.
# Create a line with Linear Regression.
# What is the slope?
# Slope is refered to as Beta
# Beta means How reactive is the stock to the market?
# Where the line intersects the vertical axis is the alpha.
# If alpha is postive, it is doing better than the market.
# Slope is not correlation
# Correlation is how tightly the points fit the line

# Scatter plots in python
# Get daily_returns
daily_returns.plot(kind='scatter', x='SPY', y='XOM')
plt.show()

beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'],1)
plt.plot(daily_returns['SPY'], beta_XOM*daily_returns['SPY'] + alpha_XOM)
daily_returns.plot(kind='scatter', x='SPY', y='XOM')

print daily_returns.corr(method='pearson')

# Real World use of Kurtosis
