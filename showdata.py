import plotly as py
import plotly.graph_objs as go

import pandas as pd
from datetime import datetime
import sys


if len(sys.argv) < 4:
    print "missisng parameters"
    sys.exit()

df = pd.read_csv(sys.argv[1])

start = int(sys.argv[2])
m = int(sys.argv[3])
total = len(df)

if start + m*2 >= total:
    print "invalid parameters"
    sys.exit()

high = max(df.HIGH[start:start+m])
low = min(df.LOW[start:start+m])
new_high = max(df.HIGH[start+m:start+2*m])
new_low = min(df.LOW[start+m:start+2*m])

name = "no-trend.html"
if new_high > high and new_low > low:
    name = "up-trend.html"
elif new_low < low and new_high < high:
    name = "down-trend.html"
else:
    pass

trace = go.Ohlc(#x=df['DTYYYYMMDD'],
                open=df.OPEN[start: start+2*m],
                high=df.HIGH[start: start+2*m],
                low=df.LOW[start: start+2*m],
                close=df.CLOSE[start: start+2*m])

data = [trace]
    #py.iplot(data, filename='simple_candlestick')
py.offline.plot(data, filename=name, auto_open=True)

