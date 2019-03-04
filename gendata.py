import plotly as py
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import pywt


def gendata(csv_file, validate_split=0.9, period_num=100):

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

def gendata21(csv_file, validate_split=0.9, period_num=100):

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

def ma_data(close, index, period, num):
    return [np.mean(close[x+1-period : x+1]) for x in range(index, index+num)]


def gendata3(csv_file, validate_split=0.9, period_num=100):

    df = pd.read_csv(csv_file)
    total = len(df)
    print 'Period:%d' % period_num
    print 'Data records:%d' % total

    datax = []
    datay = []
    ma_periods = [period_num/2, period_num, period_num*2]

    for start in range(period_num*2, total - period_num*2):

        if start % 100 == 0:
            print "%d/%d" % (start, total)

        high = max(df.CLOSE[start:start+period_num])
        low = min(df.CLOSE[start:start+period_num])
        new_high = max(df.CLOSE[start+period_num:start+2*period_num])
        new_low = min(df.CLOSE[start+period_num:start+2*period_num])

        up_trend = 0
        down_trend = 0
        no_trend = 0

        trend_mag = (high-low)*0.01

        if new_high > (high+trend_mag) and new_low > (low +trend_mag):
            up_trend = 1
        elif new_low < (low-trend_mag) and new_high < (high-trend_mag):
            down_trend = 1
        else:
            no_trend = 1

        ma = []
        for p in ma_periods:
            ma += ma_data(df.CLOSE, start, p, period_num)

        #print df.CLOSE[start:start+period_num]

        x = df.CLOSE[start:start+period_num].tolist() + ma
        high_x = max(x)
        low_x = min(x)
        mag = high_x - low_x

        if high_x == low_x:
            norm_closes = [1.0/len(x)]*len(x)
        else:
            norm_x = [(c-low_x)/mag for c in x]

        datax.append(norm_x)
        datay.append([up_trend, down_trend, no_trend])

    split_index = int(len(datax)*0.9)
    print "split at %d" % split_index

    return (np.array(datax[:split_index]), np.array(datay[:split_index])), (np.array(datax[split_index+period_num:]), np.array(datay[split_index+period_num:]))

def gendata4(csv_file, validate_split=0.9, period_num=100):

    df = pd.read_csv(csv_file)
    total = len(df)
    print 'Period:%d' % period_num
    print 'Data records:%d' % total

    datax = []
    datay = []

    (ca, cd) = pywt.dwt(df.CLOSE, "haar")
    cat = pywt.threshold(ca, np.std(ca), mode="soft")
    cdt = pywt.threshold(cd, np.std(cd), mode="soft")
    pywt_closes = pywt.idwt(cat, cdt, "haar")

    for start in range(0, total - 2*period_num):
        high = max(pywt_closes[start:start+period_num])
        low = min(pywt_closes[start:start+period_num])
        new_high = max(pywt_closes[start+period_num:start+2*period_num])
        new_low = min(pywt_closes[start+period_num:start+2*period_num])

        mag = high - low
        if mag == 0.0:
		    norm_closes = [0]*period_num
        else:
            norm_closes = [(x-low)/mag for x in pywt_closes[start:start+period_num]]

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
