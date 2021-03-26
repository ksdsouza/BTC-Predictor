from matplotlib import pyplot as plt
from numpy import genfromtxt
import pandas as pd
import datetime
df = pd.read_csv('../../../predictionVsActual.csv', sep=',',header=None, names=['Date', 'Predicted', "Actual"]).iloc[7:14]
dateNum = list(range(0, 7))
df['Day Of Year'] = dateNum
plot = df.plot(x='Day Of Year', title='Predicted Vs Actual BTC Price (Low) For W2 2019').get_figure()
plot.savefig('./figure4.pdf')
