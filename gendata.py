import plotly as py
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from operator import add
import ntpath

import plotly.graph_objs as go
from plotly.offline import plot, iplot


def gendata1(csv_file, validate_split=0.9, period_num=100):

    df = pd.read_csv(csv_file)
    total = len(df)
    print 'Period:%d' % period_num
    print 'Data records:%d' % total

    datax = []
    datay = []
    for start in range(0, total - 2*period_num):
        high = max(df.HIGH[start:start+period_num])
        low = min(df.LOW[start:start+period_num])
        new_high = max(df.HIGH[start+period_num:start+2*period_num])
        new_low = min(df.LOW[start+period_num:start+2*period_num])

        mag = high - low
        if mag == 0.0:
		    norm_opens = [0]*period_num
		    norm_highs = [0]*period_num
		    norm_lows = [0]*period_num
		    norm_closes = [0]*period_num
        else:
            norm_opens = [(x-low)/mag for x in df.OPEN[start:start+period_num]]
            norm_highs = [(x-low)/mag for x in df.HIGH[start:start+period_num]]
            norm_lows = [(x-low)/mag for x in df.LOW[start:start+period_num]]
            norm_closes = [(x-low)/mag for x in df.CLOSE[start:start+period_num]]

        up_trend = 0
        down_trend = 0
        no_trend = 0

        trend_mag = mag*0.01

        if new_high > (high+trend_mag) and new_low > (low +trend_mag):
            up_trend = 1
        elif new_low < (low-trend_mag) and new_high < (high-trend_mag):
            down_trend = 1
        else:
            no_trend = 1

        row = []
        for o,h,l,c in zip(norm_opens, norm_highs, norm_lows, norm_closes):
            row += [o,h,l,c]

        datax.append(row)
        datay.append([up_trend, down_trend, no_trend])
    split_index = int(len(datax)*0.9)
    print "split at %d" % split_index

    return (np.array(datax[:split_index]), np.array(datay[:split_index])), (np.array(datax[split_index+period_num:]), np.array(datay[split_index+period_num:]))


def gendata2(csv_file, validate_split=0.9, period_num=100):

    df = pd.read_csv(csv_file)
    total = len(df)
    print 'Period:%d' % period_num
    print 'Data records:%d' % total

    datax = []
    datay = []
    for start in range(0, total - 2*period_num):
        high = max(df.CLOSE[start:start+period_num])
        low = min(df.CLOSE[start:start+period_num])
        new_high = max(df.CLOSE[start+period_num:start+2*period_num])
        new_low = min(df.CLOSE[start+period_num:start+2*period_num])

        mag = high - low
        if mag == 0.0:
		    norm_closes = [0]*period_num
        else:
            norm_closes = [(x-low)/mag for x in df.CLOSE[start:start+period_num]]

        up_trend = 0
        down_trend = 0
        no_trend = 0

        trend_mag = mag*0.01

        if new_high > (high+trend_mag) and new_low > (low +trend_mag):
            up_trend = 1
        elif new_low < (low-trend_mag) and new_high < (high-trend_mag):
            down_trend = 1
        else:
            no_trend = 1

        datax.append(norm_closes)
        datay.append([up_trend, down_trend, no_trend])
    split_index = int(len(datax)*0.9)
    print "split at %d" % split_index

    return (np.array(datax[:split_index]), np.array(datay[:split_index])), (np.array(datax[split_index+period_num:]), np.array(datay[split_index+period_num:]))

def gen_x_data(csv_file):
    df = pd.read_csv(csv_file)
    total = len(df)
    print 'Data records:%d' % total

    datax = []
    datay = []
    ma_periods = [1, 5, 10, 20, 40,80,160, 320]

    trim = 500

    # generate ma data
    for start in range(trim , total - trim ):

        if start %10000 == 0:
            print start

        ma = []
        for i in range(start, start - 20, -2):
            for p in ma_periods:
                ma.append(np.mean(df.CLOSE[i+1-p : i+1]))

        datax.append(ma)

    a = np.asarray(datax)
    np.savetxt("x-%s"%ntpath.basename(csv_file), a, delimiter=',')

# return [1,0,0]: open buy, [0,1,0] open sell, [0,0,1]: no deal
def get_deal(high, low, start, end, price, take_profit=0.0001*10, stop_loss=0.0001*5):
    buy_closed = False
    sell_closed = False
    for i in range(start, end):
        if low[i] <= price - stop_loss:
            buy_closed = True

        if high[i] >= price + stop_loss:
            sell_closed = True

        if buy_closed and sell_closed:
            return [0,0,1,0]

        if not buy_closed and high[i] >= price + take_profit:
            return [1,0,0,0]
        elif not sell_closed and low[i] <= price - take_profit:
            return [0,1,0,0]

    return [0,0,0,1]

def gen_y_data(csv_file):

    df = pd.read_csv(csv_file)
    total = len(df)
    datay = []

    trim = 500

    for start in range(trim , total - trim):
        datay.append(df.CLOSE[start + 20])

    a = np.asarray(datay)
    np.savetxt("y-%s"%ntpath.basename(csv_file), a, delimiter=',')

def gendata(csv_file, validate_split=0.8):

    datax = np.genfromtxt("x-"+ntpath.basename(csv_file), delimiter=',');
    datay = np.genfromtxt("y-"+ntpath.basename(csv_file), delimiter=',');

    assert len(datax) == len(datay)

    split_index = int(len(datax)*validate_split)
    x = np.split(datax, [split_index])
    y = np.split(datay, [split_index])
    return (x[0], y[0]), (x[1], y[1])

def showdata(csv_file, index = 0, validate_split=0.8,  predicted=[]):

    trim = 5000
    df = pd.read_csv(csv_file)

    name = '%s-%d.html' % (csv_file, index)
    split_index = int(len(df)*validate_split)

    start = trim+split_index+index
    end = start + 100
    trace = go.Ohlc(#x=df['DTYYYYMMDD'],
                    open=df.OPEN[start: end],
	                high=df.HIGH[start: end],
	                low=df.LOW[start: end],
	                close=df.CLOSE[start: end])


    #x = [i for i range(100)]
    trace2 = go.Scatter(y=df.CLOSE[start: end])
    data = [trace, trace2]

    py.offline.plot(data, filename=name, auto_open=True)
