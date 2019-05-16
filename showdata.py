import peakutils

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import pandas as pd
from datetime import datetime

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
trace = go.Candlestick(x=df['Date'],
                open=df['AAPL.Open'],
                high=df['AAPL.High'],
                low=df['AAPL.Low'],
                close=df['AAPL.Close'])
data = [trace]

indexes = peakutils.indexes(df['AAPL.High'], thres=0.3, min_dist=30)
print(indexes)
x = [df['Date'][i] for i in indexes]
y = [df['AAPL.High'][i] for i in indexes]
print(y)

trace2 = go.Scatter(
x=x,
y=y)


data = [trace, trace2]
plot(data, filename='simple_candlestick')


