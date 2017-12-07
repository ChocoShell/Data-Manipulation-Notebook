import pandas as pd
import matplotlib.pyplot as plt

def test_run():
    # Creates a list of dates with timestamp
    start_date = '2017-01-01'
    end_date = '2017-11-01'
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    # Read SPY data
    dfSPY = pd.read_csv(
        "data/SPY.csv",
        index_col="Date", parse_dates=True,
        usecols=["Date", "Close"], na_values=['nan']
    )
    dfSPY = dfSPY.rename(columns={'Close': "SPY"})

    # Joining into df
    df = df.join(dfSPY, how='inner')

    # Read in more stocks
    symbols = ['GOOG', 'FB', 'NFLX']
    for symbol in symbols:
        df_temp = pd.read_csv(
            "data/{}.csv".format(symbol),
            index_col='Date', parse_dates=True,
            usecols=['Date', 'Close'],
            na_values=['nan'])

        # Rename to prevent clash
        df_temp = df_temp.rename(columns={'Close': symbol})
        df = df.join(df_temp, how='inner')
        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])

        # Slice by row range (dates).
        # Won't work the other way around.
        # print df.ix['2016-11-01':'2017-11-01']

        # # Column Slicing
        # print df
    

        # # Slice by both
        # print df.ix['2016-11-01':'2017-11-01', ['GOOG', 'FB']]

        # Normalizes everything according to the first row
        # and faster than for loop
        df = df / df.ix[0, :]

    print df.ix[:, ['GOOG', 'FB']]
    print df.ix['2017-10-01':'2017-11-01', ['GOOG', 'FB']]
    #ix: A primarily label-location based indexer, with integer position fallback.
    plot_data(df)

def plot_data(df, title='Stock Prices'):
    '''Plot Data'''
    ax = df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_label("Date")
    plt.show()


if __name__ == "__main__":
    test_run()
