import plotly as py
import plotly.graph_objs as go

import pandas as pd
from datetime import datetime

def gendata(csv_file, validate_split=0.9, period_num=100):

    df = pd.read_csv(csv_file)
    total = len(df)
    print 'Period:%d' % period_num
    print 'Data records:%d' % total

    for start in range(0, total - 2*period_num):
        high = max(df.HIGH[start:start+period_num])
        low = min(df.LOW[start:start+period_num])
        new_high = max(df.HIGH[start+m:start+2*period_num])
        new_low = min(df.LOW[start+m:start+2*period_num])

        mag = high - low

        norm_opens = [(x-low)/mag for x in df.OPEN[start:start+period_num]]
        norm_highs = [(x-low)/mag for x in df.HIGH[start:start+period_num]]
        norm_lows = [(x-low)/mag for x in df.LOW[start:start+period_num]]
        norm_closes = [(x-low)/mag for x in df.CLOSE[start:start+period_num]]

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

