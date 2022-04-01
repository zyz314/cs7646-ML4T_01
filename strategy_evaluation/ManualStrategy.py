import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from util import get_data
from indicators import getSMA, getBollingerBand, getMomentum

class ManualStrategy(object):

    def author(self):
        return 'fyuen3'



    def testPolicy(
            self,
            symbol="JPM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=100000):
        prices_df = get_data([symbol], pd.date_range(sd, ed))
        prices_df.ffill(inplace=True)
        prices_df.bfill(inplace=True)
        if 'SPY' != symbol:
            prices_df.drop(labels='SPY', axis=1, inplace=True)

        holding_df = pd.DataFrame(index=prices_df.index, data=np.zeros(prices_df.shape), columns=prices_df.columns)
        df_trades = pd.DataFrame(index=prices_df.index, columns=['Trades'])
        lookback = 14
        sma, sma_50_days, price_over_sma = getSMA(prices_df, lookback)
        top_band, bottom_band, bbp = getBollingerBand(prices_df, lookback)
        momentum = getMomentum(prices_df, lookback)
        for i in range(holding_df.shape[0]):
            index = holding_df.index[i]
            price_change = 0
            if price_over_sma.iloc[i][symbol] < 0.6 or bbp.iloc[i][symbol] < 0.2 or momentum.iloc[i][symbol] < -0.1:
                price_change = 1
            elif price_over_sma.iloc[i][symbol] > 1.0 or bbp.iloc[i][symbol] > 0.8 or momentum.iloc[i][symbol] > -0.1:
                price_change = -1

            holding = 0
            if i > 0:
                holding = holding_df.iloc[i - 1][symbol]
            position = np.sign(price_change) * 1000
            if holding + position > 1000 or holding + position < -1000:
                position = 0
            elif 1000 >= holding + position * 2 >= -1000:
                position = position * 2

            df_trades.loc[index, 'Trades'] = position
            holding_df.iloc[i][symbol] = position + holding
        df_trades.iloc[-1] = 0
        return df_trades

