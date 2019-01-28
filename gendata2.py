import plotly as py
import plotly.graph_objs as go

import pandas as pd
from datetime import datetime
import sys

if len(sys.argv) < 3 :
    print "missisng parameters"
    sys.exit()

df = pd.read_csv(sys.argv[1])

m = int(sys.argv[2])
total = len(df)
print 'Period:%d' % m
print 'Total:%d' % total

fx = open(sys.argv[1].split('.')[0] + "-x.csv", "w");
fy = open(sys.argv[1].split('.')[0] + "-y.csv", "w");

for start in range(0, total - int(1.5*m)):
    high = max(df.HIGH[start:start+m])
    low = min(df.LOW[start:start+m])
    new_high = max(df.HIGH[start+m:start+int(1.5*m)])
    new_low = min(df.LOW[start+m:start+int(1.5*m)])

    mag = high - low

    norm_opens = [(x-low)/mag for x in df.OPEN[start:start+m]]
    norm_highs = [(x-low)/mag for x in df.HIGH[start:start+m]]
    norm_lows = [(x-low)/mag for x in df.LOW[start:start+m]]
    norm_closes = [(x-low)/mag for x in df.CLOSE[start:start+m]]

    up_trend = 0
    down_trend = 0
    no_trend = 0

    if new_high > high and new_low > low:
        up_trend = 1
    elif new_low < low and new_high < high:
        down_trend = 1
    else:
        no_trend = 1

    first = True
    for o,h,l,c in zip(norm_opens, norm_highs, norm_lows, norm_closes):
        if first:
            fx.write("%.4f,%.4f,%.4f,%.4f" % (o,h,l,c))
            first = False
        else:
            fx.write(",%.4f,%.4f,%.4f,%.4f" % (o,h,l,c))

    fx.write("\n")

    fy.write("%d,%d,%d\n"%(up_trend, down_trend, no_trend))

fx.close()
fy.close()

