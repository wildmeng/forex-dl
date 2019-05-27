
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import pandas as pd
from datetime import datetime
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

df = df.truncate(before=80,after=120)
print(len(df))

#closes = df['AAPL.Close'].values[:100]


#datax = np.reshape(df['AAPL.Close'].values, (1, 20))

#print(datax)

trace = go.Candlestick(x=df['Date'],
                open=df['AAPL.Open'],
                high=df['AAPL.High'],
                low=df['AAPL.Low'],
                close=df['AAPL.Close'])
data = [trace]
data = [trace]
plot(data, filename='simple_candlestick')


