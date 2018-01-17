# TODO plot portfolio

import pandas as pd
from utils import get_data, plot_data, stock_nearest_neighbors, KNNLearner, LinRegLearner
import scipy.optimize

import numpy as np

def find_portfolio_statistics(allocs, df):

    dfcopy = df.copy()

    # Find cumulative value over time
    df = (df/df.ix[0])
    df = df * allocs
    df = df.sum(axis=1)

    # Compute Portfolio Statistics
    cumulative_return = (df.ix[-1]/df.ix[0]) - 1
    dailyreturns = (df.ix[1:]/df.ix[:-1].values) - 1
    average_daily_return = dailyreturns.mean(axis=0)
    std_daily_return = dailyreturns.std(axis=0)
    sharpe_ratio = (252**(1/2.0)) * ((average_daily_return-0)/std_daily_return)
    starting_value = df.ix[0]
    allocations = [np.around(allocs[i], 2) for i in range(len(allocs))]

    ending_value = df.ix[-1]
    total_returns = dailyreturns.sum(axis=0)
    return {
        "allocations": allocations,
        "cumulative_return":np.around(100*cumulative_return,2),
        "dailyreturns":dailyreturns,
        "average_daily_return":np.around(100*average_daily_return,2),
        "std_daily_return":np.around(100*std_daily_return,2),
        "sharpe_ratio":np.around(sharpe_ratio,2),
        "ending_value":np.around(100*ending_value,2), 
        "total_returns":np.around(100*total_returns,2)
    }


def sharpe(allocs, df):

    dfcopy = df.copy()

    # Find cumulative value over time
    dfcopy = (dfcopy/dfcopy.ix[0])
    dfcopy = dfcopy * allocs
    dfcopy = dfcopy.sum(axis=1)

    dailyreturns = (dfcopy.ix[1:]/dfcopy.ix[:-1].values) - 1
    average_daily_return = dailyreturns.mean(axis=0)
    std_daily_return = dailyreturns.std(axis=0)
    sharpe_ratio = (252**(1/2.0)) * ((average_daily_return-0)/std_daily_return)
    return -1.0 * sharpe_ratio

def optimize_allocations(df):
    initial_guess = []
    bounds = ()
    l = list(bounds)
    for column in df:
        initial_guess.append(1/float(df.shape[1]))
        l.append((0.0,1.0))

    cons = ({'type': 'eq', 'fun': lambda x:  sum(x) - 1})

    max_result = scipy.optimize.minimize(
        sharpe,
        initial_guess,
        args=(df,),
        method='SLSQP',
        bounds=tuple(l),
        constraints=cons
    )

    return max_result

def normalize(df):
    dfcopy = df.copy()
    return dfcopy / dfcopy.ix[0, :]

def main():
    # List of symbols to plot
    symbols = ['SPY', 'FB', 'NFLX', 'GOOG']

    # Creates a list of dates with timestamp
    start_date = '01-Dec-2016'
    end_date = '28-Nov-17'
    dates = pd.date_range(start_date, end_date)

    df = get_data(symbols, dates)

    optimized_allocs = optimize_allocations(df)

    port_stats = find_portfolio_statistics(optimized_allocs.x, df)

    print('Portfolio Statistics')

    for i in range(len(optimized_allocs.x)):
        print('Allocation for {}: {}%'.format(symbols[i], 100*port_stats['allocations'][i]))
        print('Cumulative Return for {}: {}%'.format(symbols[i], np.around(100*optimized_allocs.x[i] * (df[symbols[i]][-1] - df[symbols[i]][0])/df[symbols[i]][0], 2)))
    print('\nCumulative Return: {}%'.format(port_stats['cumulative_return']))
    print('Sum of Allocations: {}'.format( np.around(sum(port_stats['allocations']),2)))
    # port_stats["dailyreturns"]
    print('average_daily_return: {}%'.format(port_stats["average_daily_return"]))
    print('std_daily_return: {}%'.format(port_stats["std_daily_return"]))
    print('sharpe_ratio: {}'.format(port_stats["sharpe_ratio"]))
    print('ending_value: {}%'.format(port_stats["ending_value"]))
    print('total_returns: {}%'.format(port_stats["total_returns"]))

    dfport = df * optimized_allocs.x
    dfport =  pd.DataFrame(dfport.sum(axis=1), columns=['Portfolio'])
    df = df.join(dfport)
    # plot_data(normalize(df))
    x = np.array(dfport.index.values)
    x_ = np.reshape(x, (-1,1))
    y = np.array(dfport.values)
    y = np.reshape(y, (-1,1))

    knn = KNNLearner(3)
    knn.train(dfport)
    y_ = knn.query(x_)

    def interpolate_delta(df, inplace=False):
        if not inplace:
            df = df.copy()
        ind = df.index
        df.index = df.index.total_seconds()
        df.interpolate(method="index", inplace=True)
        df.index = ind
        return df

    linreg = LinRegLearner()
    linreg.train(dfport)
    print type(x[0])
    print type(np.timedelta64(1, 's'))

    helper = np.vectorize(lambda x: x.total_seconds)
    #x = interpolate_delta(x)
    y_lin = linreg.query(helper(x))


    

    import matplotlib.pyplot as plt
    plt.plot(x, y_,'b-', c='g', label='KNN')
    plt.plot(x, y_lin,'b-', c='g', label='LinReg')

    plt.show()

if __name__ == "__main__":
    main()
