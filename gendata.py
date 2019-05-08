import plotly as py
import plotly.graph_objs as go

import pandas as pd
import numpy as np


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
    ma_periods = [1] + [i*5 for i in range(1,81)]

    trim = 500

    # generate ma data
    for start in range(trim , total - trim ):
        ma = []
        for p in ma_periods:
            ma.append(np.mean(df.CLOSE[start+1-p : start+1]))

        x = ma
        high_x = max(x)
        low_x = min(x)
        mag = high_x - low_x

        if high_x == low_x:
            norm_x = [1.0/len(x)]*len(x)
        else:
            norm_x = [(c-low_x)/mag for c in x]

        datax.append(norm_x)

    a = np.asarray(datax)
    np.savetxt("x-%s"%csv_file, a, delimiter=',')

# return [1,0,0]: open buy, [0,1,0] open sell, [0,0,1]: no deal
def get_deal(high, low, start, end, price, take_profit=0.0001*100, stop_loss=0.0001*50):
    buy_closed = False
    sell_closed = False
    for i in range(start, end):
        if low[i] <= price - stop_loss:
            buy_closed = True

        if high[i] >= price + stop_loss:
            sell_closed = True

        if not buy_closed and high[i] >= price + take_profit:
            return [1,0,0]
        elif not sell_closed and low[i] <= price - take_profit:
            return [0,1,0]

    return [0,0,1]

def gen_y_data(csv_file):

    df = pd.read_csv(csv_file)
    total = len(df)
    datay = []

    trim = 500

    for start in range(trim , total - trim):
        result = get_deal(df.HIGH, df.LOW, start+1, start+1 + 200, df.CLOSE[start])
        datay.append(result)

    a = np.asarray(datay)
    np.savetxt("y-%s"%csv_file, a, delimiter=',')

def gendata(csv_file, validate_split=0.8):

    datax = np.genfromtxt("x-"+csv_file, delimiter=',');
    datay = np.genfromtxt("y-"+csv_file, delimiter=',');

    assert len(datax) == len(datay)

    split_index = int(len(datax)*validate_split)
    x = np.split(datax, [split_index])
    y = np.split(datay, [split_index])
    return (x[0], y[0]), (x[1], y[1])

def showdata(csv_file, index , trends, validate_split=0.9, period_num=100):

	df = pd.read_csv(csv_file)
	split_index = int(len(df)*0.9)

	start = split_index + index
	m = period_num
	total = len(df)

	if start + m*2 >= total:
	    print "invalid parameters"
	    sys.exit()

	high = max(df.HIGH[start:start+m])
	low = min(df.LOW[start:start+m])
	new_high = max(df.HIGH[start+m:start+2*m])
	new_low = min(df.LOW[start+m:start+2*m])

	name = "no-trend"
	if new_high > high and new_low > low:
	    name = "up-trend"
	elif new_low < low and new_high < high:
	    name = "down-trend"
	else:
	    pass

	name += '--up:%.4f-down:%.4f-swing:%0.4f.html' % (trends[0], trends[1], trends[2])

	trace = go.Ohlc(#x=df['DTYYYYMMDD'],
	                open=df.OPEN[start: start+2*m],
	                high=df.HIGH[start: start+2*m],
	                low=df.LOW[start: start+2*m],
	                close=df.CLOSE[start: start+2*m])

	data = [trace]
	    #py.iplot(data, filename='simple_candlestick')
	py.offline.plot(data, filename=name, auto_open=True)
