import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from util import get_data


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "fyuen3"


def getSMA(prices, lookback):
    # calculate simple moving average from prices
    sma = prices.rolling(window=lookback, min_periods=lookback).mean()
    sma_50_days = prices.rolling(window=50, min_periods=50).mean()
    cross_signal_df = sma / sma_50_days

    price_over_sma = prices / sma - 1
    return sma, sma_50_days, price_over_sma


def getBollingerBand(prices, lookback):
    sma, _, _ = getSMA(prices, lookback)
    rolling_std = prices.rolling(window=lookback, min_periods=lookback).std()

    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)

    bbp = (prices - bottom_band) / (top_band - bottom_band)
    return top_band, bottom_band, bbp


def getMomentum(prices, lookback):
    momentum = prices / prices.shift(lookback) - 1
    return momentum

def getMACD(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def getPPO(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    ppo = ema12 - ema26 / ema26
    signal = ppo.ewm(span=9, adjust=False).mean()
    return ppo, signal

def plotGraph(symbol,sd, ed):
    prices_df = get_data([symbol], pd.date_range(sd, ed))
    prices_df.ffill(inplace=True)
    prices_df.bfill(inplace=True)
    if 'SPY' != symbol:
        prices_df.drop(labels='SPY', axis=1, inplace=True)

    normed_price = prices_df / prices_df.iloc[0]

    lookback = 20
    sma, sma_50_days, cross_signal_df = getSMA(normed_price,lookback)

    # figure 1.
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set(xlabel='Date', ylabel="Price", title="Simple Moving Average")
    ax.plot(normed_price, "red", label="Normalized Price")
    ax.plot(sma, "blue", label="20-day SMA")
    ax.plot(sma_50_days, "green", label="50-day SMA")

    ax.legend(loc="best")
    fig.savefig('Indicator1_SMA.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set(xlabel='Date', ylabel="Price", title="Cross Signal")
    ax.plot(normed_price, "red", label="Normalized Price")
    ax.plot(cross_signal_df, "green", label="20-day / 50-day SMA")

    ax.legend(loc="best")
    fig.savefig('Indicator1_cross_SMA.png')
    plt.close()

    top_band, bottom_band, bbp = getBollingerBand(normed_price, lookback)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set(xlabel='Date', ylabel="Price", title="Bollinger Bands")
    ax.plot(normed_price, "red", label='Normalized Price')
    ax.plot(sma, "blue", label="Moving Average")
    ax.plot(top_band, "green", label="Upper Band")
    ax.plot(bottom_band, "yellow", label="Lower Band")
    ax.legend()
    fig.savefig('Indicator2_BollingerBand.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set(xlabel='Date', ylabel="Price", title="Bollinger Bands")
    ax.plot(normed_price, "red", label='Normalized Price')
    ax.plot(bbp, "green", label="Upper Band")
    ax.legend()
    fig.savefig('Indicator2_BBP.png')
    plt.close()


    momentum = getMomentum(normed_price, lookback)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set(xlabel='Date', ylabel="Price", title="Momentum")
    ax.plot(normed_price, "red", label='Normalized Price')
    ax.plot(momentum, "blue", label="Momentum")
    ax.legend()
    fig.savefig('Indicator3_Momentum.png')
    plt.close()

    macd, signal = getMACD(normed_price)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set(xlabel='Date', ylabel="Price", title="MACD")
    ax.plot(normed_price, "red", label='Normalized Price')
    ax.plot(macd, "blue", label="MACD")
    ax.plot(signal, "green", label="MACD Signal")
    ax.legend()
    fig.savefig('Indicator4_MACD.png')
    plt.close()

    ppo, signal = getMACD(normed_price)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set(xlabel='Date', ylabel="Price", title="Percentage Price Indicator")
    ax.plot(normed_price, "red", label='Normalized Price')
    ax.plot(ppo, "blue", label="PPO")
    ax.plot(signal, "green", label="PPO Signal")
    ax.legend()
    fig.savefig('Indicator5_PPO.png')
    plt.close()