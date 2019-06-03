from datetime import datetime
import backtrader as bt
from buy_strategy import MyStrategy

class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=1), bt.ind.SMA(period=60)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(MyStrategy, period=100)

    #cerebro.addstrategy(SmaCross)

    # sh: 000001.SS
    # BT: BTC-USD
    # SP500: ^GSPC
    data0 = bt.feeds.YahooFinanceData(dataname='^GSPC', fromdate=datetime(2013, 1, 1),
                                      todate=datetime(2019, 6, 1), decimals=5)
    cerebro.adddata(data0)

    # Set our desired cash start
    cerebro.broker.setcash(10000.0)

    # Add a FixedSize sizer according to the stake
    #cerebro.addsizer(bt.sizers.FixedSize, stake=2)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=90)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot(style='bar')