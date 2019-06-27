import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pandas as pd
from sklearn import preprocessing
from matplotlib.finance import candlestick_ohlc

def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

csv_file = "./data/DAT_ASCII_EURUSD_M1_2018.csv"
df = pd.read_csv(csv_file, sep=';')
index = 0
x = df.close[index: index+20]
x = preprocessing.minmax_scale(x)
print(x)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
l, = plt.plot(np.arange(20), x, '--bo',lw=2, color='red')
#plt.axis([0, 1, -10, 10])

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=0)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=0)

plt.autoscale(enable=True, axis="y", tight=True)

def update(val):
    autoscale_y(ax)
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()
sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    global index
    index += 1
    x = df.close[index: index+20]
    x = preprocessing.minmax_scale(x)
    l.set_ydata(x)
    autoscale_y(ax)
    fig.canvas.draw_idle()


button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    #plt.axis([0, 1, -20, 20])
    ax.set_ylim(0,1)
    l.set_color(label)
    fig.canvas.draw_idle()

radio.on_clicked(colorfunc)

plt.show()
