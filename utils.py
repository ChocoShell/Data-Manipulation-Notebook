import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
import numpy as np

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

def stock_nearest_neighbors(data, n=3):
    T = np.array(data.index.values)
    T = np.reshape(T, (-1,1))
    print(data)

    for i, weights in enumerate(['uniform', 'distance']):
        print(weights)
        knn = neighbors.KNeighborsRegressor(n, weights=weights)
        
        x = np.array(data.index.values)
        x = np.reshape(x, (-1,1))
        y = np.array(data.values)
        y = np.reshape(y, (-1,1))
        # import ipdb; ipdb.set_trace()
        y_ = knn.fit(x, y).predict(x)

        plt.subplot(2, 1, i + 1)
        plt.plot(x, y, 'r-', label='data')
        plt.plot(x, y_,'b-', c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n,
                                                                weights))
    plt.show()

class KNNLearner:
    def __init__(self, k):
        self.k = k
        
    def train(self, data):
        knn = neighbors.KNeighborsRegressor(self.k, weights='uniform')
        x = np.array(data.index.values)
        x = np.reshape(x, (-1,1))
        y = np.array(data.values)
        y = np.reshape(y, (-1,1))
        self.algo = knn.fit(x, y)

    def query(self, x):
        return self.algo.predict(x)

class LinRegLearner:
    def __init__(self):
        pass
        
    def train(self, data):
        linreg = LinearRegression()
        x = np.array(data.index.values)
        x = np.reshape(x, (-1,1))
        y = np.array(data.values)
        y = np.reshape(y, (-1,1))
        
        def interpolate_delta(df, inplace=False):
            if not inplace:
                df = df.copy()
            ind = df.index
            df.index = df.index.total_seconds()
            df.interpolate(method="index", inplace=True)
            df.index = ind
            return df
        #x = interpolate_delta(data).index
        print x
        self.algo = linreg.fit(x, y) # from scipy or numpy
    
    def query(self, x):
        return self.algo.predict(x[:, np.newaxis])
        
