import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import marketsimcode as msc
from util import get_data
from indicators import getSMA, getBollingerBand, getMomentum


def author():
    return 'fyuen3'


def testPolicy(
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
    df_trades['Trades'] = 0

    lookback = 14
    sma, sma_50_days, price_over_sma = getSMA(prices_df, lookback)
    top_band, bottom_band, bbp = getBollingerBand(prices_df, lookback)
    momentum = getMomentum(prices_df, lookback)

    signal = 0

    for i in range(prices_df.shape[0]):
        index = prices_df.index[i]
        if signal == 0:
            if price_over_sma.iloc[i][symbol] < 0.6 or bbp.iloc[i][symbol] < 0.2 or momentum.iloc[i][symbol] < -0.1:
                df_trades.loc[index, 'Trades']= 1000
                signal = 1

            elif price_over_sma.iloc[i][symbol] > 1.0 or bbp.iloc[i][symbol] > 0.8 or momentum.iloc[i][symbol] > 0.1:
                df_trades.loc[index, 'Trades']= -1000
                signal = -1

        elif signal == -1:
            if price_over_sma.iloc[i][symbol] < 0.6 or bbp.iloc[i][symbol] < 0.2 or momentum.iloc[i][symbol] < -0.1:
                df_trades.loc[index, 'Trades']= 2000
                signal = 1

        elif signal == 1:
            if price_over_sma.iloc[i][symbol] > 1.3 or bbp.iloc[i][symbol] > 0.8 or momentum.iloc[i][symbol] > 0.2:
                df_trades.loc[index, 'Trades']= -2000
                signal = -1
    df_trades.iloc[-1] = 0
    return df_trades


def benchmark(symbol, sd, ed, shares):
    prices_df = get_data([symbol], pd.date_range(sd, ed))
    df_trades = pd.DataFrame(index=prices_df.index, columns=['Trades'])
    df_trades['Trades'] = 0

    start = df_trades.index.min()
    df_trades.loc[start, 'Trades'] = shares
    return df_trades


def plot_graph(symbol, sd, ed, df_trades, df_trades_benchmark, in_sample=False):
    df_orders = df_trades.copy()
    df_orders['Symbol'] = symbol
    df_orders['Order'] = np.where(df_orders['Trades'] > 0, 'BUY', 'SELL')
    df_orders['Shares'] = np.abs(df_orders['Trades'])

    port_vals = msc.compute_portvals(df_orders, start_val=100000, commission=9.5, impact=0.005)
    df_orders = df_trades_benchmark.copy()
    df_orders['Symbol'] = symbol
    df_orders['Order'] = np.where(df_orders['Trades'] > 0, 'BUY', 'SELL')
    df_orders['Shares'] = np.abs(df_orders['Trades'])

    benchmark_port_vals = msc.compute_portvals(df_orders, start_val=100000, commission=9.5, impact=0.005)
    normed_port = port_vals / port_vals.iloc[0]
    normed_bench = benchmark_port_vals / benchmark_port_vals.iloc[0]

    plt.figure(figsize=(12, 6.5))
    plt.plot(normed_port, label="Portfolio", color='red', lw=2)
    plt.plot(normed_bench, label="Benchmark", color='purple', lw=1.2)

    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)

    for index, marks in df_trades.iterrows():
        if df_trades.loc[index, 'Trades'] > 0:
            plt.axvline(x=index, color='blue', linestyle='dashed', alpha=.9)
        elif df_trades.loc[index, 'Trades'] < 0:
            plt.axvline(x=index, color='black', linestyle='dashed', alpha=.9)
    if in_sample:
        plt.title('In Sample vs Benchmark')
        plt.savefig('in_sample.png')
    else:
        plt.title('Out of Sample vs Benchmark')
        plt.savefig('out_sample.png')
    plt.close()

    # Print statistics
    print(f"Start Date: {sd}")
    print(f"End Date: {ed}")
    print(f"Symbol: {symbol}")

    port_cr, port_adr, port_stddr, port_sr = msc.assess_portfolio(port_vals)
    bench_cr, bench_adr, bench_stddr, bench_sr = msc.assess_portfolio(benchmark_port_vals)

    print(f"Cumulative return of the benchmark: {bench_cr}")
    print(f"Cumulative return of the portfolio: {port_cr}")

    print(f"Stdev of daily returns of benchmark: {bench_stddr}")
    print(f"Stdev of daily returns of portfolio: {port_stddr}")

    print(f"Mean of daily returns of benchmark: {bench_adr}")
    print(f"Mean of daily returns of portfolio: {port_adr}")


def plot_in_sample():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = 'JPM'
    df_trades = testPolicy(symbol=symbol, sd=sd, ed=ed,
                                          sv=100000)
    df_trades_benchmark = benchmark(symbol=symbol, sd=sd, ed=ed,
                                                   shares=1000)
    plot_graph(symbol=symbol, sd=sd, ed=ed, df_trades=df_trades,
               df_trades_benchmark=df_trades_benchmark, in_sample=True)


def plot_out_of_sample():

    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    symbol = 'JPM'
    df_trades = testPolicy(symbol=symbol, sd=sd, ed=ed,
                           sv=100000)
    df_trades_benchmark = benchmark(symbol=symbol, sd=sd, ed=ed,
                                    shares=1000)
    plot_graph(symbol=symbol, sd=sd, ed=ed, df_trades=df_trades,
               df_trades_benchmark=df_trades_benchmark, in_sample=False)

