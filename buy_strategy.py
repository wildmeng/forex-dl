
# Import the backtrader platform
import numpy as np
import backtrader as bt
from sklearn import preprocessing
from model import get_model

class TrendIndictor(bt.Indicator):
    lines = ('up','down', 'flat','down2flat', 'up2flat',
             'flat2up', 'flat2down',  'up2down','down2up')
    params = dict(period=30, model=None)
    plotlines = dict(flat=dict(ls='--'))

    def __init__(self):
        self.addminperiod(self.params.period)
        #self.model = get_model(self.params.period)

    def next(self):

        datax = self.data.get(size=self.p.period)

        x = np.array(datax)
        x =np.reshape(x, (1,self.p.period))
        x = preprocessing.scale(x, axis=1)
        p = self.p.model.predict(x)

        trend = p[0].tolist()

        self.lines.up[0] = trend[0]
        self.lines.down[0] = trend[1]
        self.lines.flat[0] = trend[2]

        self.lines.down2flat[0] = trend[3]
        self.lines.up2flat[0] = trend[4]
        self.lines.flat2up[0] = trend[5]
        self.lines.flat2down[0] = trend[6]

        self.lines.up2down[0] = trend[7]
        self.lines.down2up[0] = trend[8]

# Create a Stratey
class MyStrategy(bt.Strategy):
    params = (
        ('period', 10),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.isbuy = True
        self.model = get_model(self.params.period)
        self.short = TrendIndictor(period=self.p.period, model=self.model)
        self.mid = TrendIndictor(self.data1, period=self.p.period, model=self.model)
        self.long = TrendIndictor(self.data2, period=self.p.period, model=self.model)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def get_up_trend(self, trend):
        up_trend = trend.l.down2up + trend.l.up + trend.l.flat2up
        return up_trend

    def get_down_trend(self, trend):
        down_trend = trend.l.up2down + trend.l.down + trend.l.flat2down
        return down_trend

    def open_order(self):
        # we MIGHT BUY if ...
        if self.position:
            return

        thres = 0.8
        # try to open long
        if self.get_up_trend(self.short) > thres and self.get_down_trend(self.mid) < (1-thres):
            self.log('Long, %.2f' % (self.dataclose[0]))
            # Keep track of the created order to avoid a 2nd order
            self.order = self.buy()
            self.isbuy = True
            return

    def close_order(self):
        thres = 0.8
        if self.isbuy:
            #if self.get_down_trend(self.long) > thres:
            if self.get_down_trend(self.short) > thres:
                self.log('Close Long order, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.close()

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            self.open_order()
        else:
            # Already in the market
            self.close_order()
            self.open_order()
