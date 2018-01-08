import pandas as pd
import matplotlib.pyplot as plt

def get_data(symbols, dates, normal=False):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv("data/{}.csv".format(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Close'], na_values=['nan'])

        df_temp = df_temp.rename(columns={'Close': symbol})
        df = df.join(df_temp)
        df = df.dropna()

        # Normalizes everything according to the first row
        # and faster than for loop
        if normal:
            df = df / df.ix[0, :]

    return df

def plot_data(df, title='Stock Prices'):
    '''Plot Data'''
    ax = df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_label("Date")
    plt.grid()
    plt.show()
