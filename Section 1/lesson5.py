# Lesson Outline https://classroom.udacity.com/courses/ud501/lessons/4156938722/concepts/45439393860923
import pandas as pd
import matplotlib.pyplot as plt

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv("data/{}.csv".format(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

# Global Statistics
def test_run():
    dates = pd.date_range('2016-11-01', '2017-11-01')
    symbols = ['SPY', 'FB', 'GOOG', 'NFLX']
    df = get_data(symbols, dates)
    print df
    df.plot()
    plt.show()
    # plot_data(df)

    # Compute global stats for each stock
    print df.mean()
    print df.median()
    print df.std()

# Rolling Statistics - Taking a snapshot over windows (certain ranges)
#  20 day simple moving average / rolling mean - Technical Indicator
#  When does price cross SMA? Hypothesis - rolling mean is seen as "true" stock price
# Rolling STD can tell us if the stock change is significant
# Bollinger Bands - See the volatility.  If it is not very volatile, then maybe it might work.
# 2 STD means you should pay attention.

def test_run2():
    dates = pd.date_range('2010-01-01', '2012-12-31')
    symbols = ['SPY', 'FB', 'GOOG', 'NFLX']
    df = get_data(symbols, dates)

    # PLot SPY data, retain matplotlib axis object
    ax = df['SPY'].plot(title="SPY rolling mean", label='SPY')

    # Compute rolling mean using 20 day window
    rm_SPY = pd.rolling_mean(df['SPY'], window=20)
    # Non deprecated version
    pd.rolling(values, window=window).mean()


    def get_rolling_std(values, window):
        """Return rolling standard deviation of given values, using specified window size."""
        # TODO: Compute and return rolling standard deviation
        return pd.rolling(values, window=window).std()

    # ax=ax adds new plot to old plot
    rm_SPY.plot(label='Rolling mean', ax=ax)

    #labels
    ax.add_xlabel("Date")
    ax.add_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

    # Compute Bollinger Bands
    # 1. Computer rolling mean
    # 2. Compute rolling standard deviation
    # 3. Computer upper_band and lower_band
    # NON DEPRECATED
    def get_rolling_mean(values, window):
        """Return rolling mean of given values, using specified window size."""
        return pd.rolling(values, window=window).mean()


    def get_rolling_std(values, window):
        """Return rolling standard deviation of given values, using specified window size."""
        # TODO: Compute and return rolling standard deviation
        return pd.rolling(values, window=window).std()


    def get_bollinger_bands(rm, rstd):
        """Return upper and lower Bollinger Bands."""
        # TODO: Compute upper_band and lower_band
        upper_band = rm + 2*rstd
        lower_band = rm - 2*rstd
        return upper_band, lower_band

# Daily Returns
# daily_ret[t] = (price[t]/price[t-1])-1

def compute_daily_returns(df):
    daily_ret = df.copy()
    
    daily_ret[1:] = (df[1:]/df[:-1].values) -1
    # OR
    daily_ret = (df/df.shift(1)) - 1 # better pandas way

    daily_ret.ix[0,:] = 0

# Cumulative returns
    cum_ret[t] = (price[t]/price[0])-1
    # For the year, t is the last day of the year.
    cum_ret = (price/price[0])-1


def plot_data(df, title='Stock Prices'):
    '''Plot Data'''
    ax = df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_label("Date")
    plt.show()


if __name__ == "__main__":
    test_run()
