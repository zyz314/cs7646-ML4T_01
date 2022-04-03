import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import ManualStrategy as ms
import marketsimcode as msc

import StrategyLearner as sl
from util import get_data

def author():
    return 'fyuen3'

def experiment1():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    plot_graph(sd=sd, ed=ed, in_sample=True)
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    plot_graph(sd=sd, ed=ed, in_sample=False)

def plot_graph(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), in_sample=True):
    symbol = 'JPM'
    df_trades = ms.testPolicy(symbol=symbol, sd=sd, ed=ed,
                                          sv=100000)
    df_trades_benchmark = sl.benchmark(symbol=symbol, sd=sd, ed=ed,
                                                   shares=1000)
    df_orders = df_trades.copy()
    df_orders['Symbol'] = symbol
    df_orders['Order'] = np.where(df_orders['Trades'] > 0, 'BUY', 'SELL')
    df_orders['Shares'] = np.abs(df_orders['Trades'])

    port_vals = msc.compute_portvals(df_orders, start_val=100000, commission=9.95, impact=0.005)
    df_orders = df_trades_benchmark.copy()
    df_orders['Symbol'] = symbol
    df_orders['Order'] = np.where(df_orders['Trades'] > 0, 'BUY', 'SELL')
    df_orders['Shares'] = np.abs(df_orders['Trades'])

    benchmark_port_vals = msc.compute_portvals(df_orders, start_val=100000, commission=9.5, impact=0.005)
    normed_port = port_vals / port_vals.iloc[0]
    normed_bench = benchmark_port_vals / benchmark_port_vals.iloc[0]

    port_cr, port_adr, port_stddr, port_sr = msc.assess_portfolio(port_vals)
    bench_cr, bench_adr, bench_stddr, bench_sr = msc.assess_portfolio(benchmark_port_vals)

    learner = sl.StrategyLearner(impact=0.005)

    learner.add_evidence(symbol = symbol, sd=sd, ed=ed, sv=100000)
    df_trades_sl = learner.testPolicy(symbol=symbol, sd=sd, ed=ed,
                                          sv=100000)

    df_orders_sl = df_trades_sl.copy()
    df_orders_sl['Symbol'] = symbol
    df_orders_sl['Order'] = np.where(df_orders_sl['Trades'] > 0, 'BUY', 'SELL')
    df_orders_sl['Shares'] = np.abs(df_orders_sl['Trades'])

    port_vals_sl = ms.compute_portvals(df_orders_sl, start_val=100000, commission=9.5, impact=0.005)

    normed_port_sl = port_vals_sl / port_vals_sl.iloc[0]

    sl_cr, sl_adr, sl_stddr, sl_sr = ms.assess_portfolio(port_vals_sl)
    print(f"Cumulative return of the benchmark: {bench_cr}")
    print(f"Cumulative return of the Manual Strategy portfolio: {port_cr}")
    print(f"Cumulative return of the Strategy Learner portfolio: {sl_cr}")

    print(f"Stdev of daily returns of benchmark: {bench_stddr}")
    print(f"Stdev of daily returns of Manual Strategy portfolio: {port_stddr}")
    print(f"Stdev of daily returns of Strategy Learner portfolio: {sl_stddr}")

    print(f"Mean of daily returns of benchmark: {bench_adr}")
    print(f"Mean of daily returns of Manual Strategy portfolio: {port_adr}")
    print(f"Mean of daily returns of Strategy Learner portfolio: {sl_adr}")


    plt.figure(figsize=(12, 6.5))
    plt.plot(normed_port, label="Manual Strategy Portfolio", color='red', lw=2)
    plt.plot(normed_port_sl, label="Strategy Learner Portfolio", color='green', lw=2)
    plt.plot(normed_bench, label="Benchmark", color='purple', lw=1.2)

    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    if in_sample :
        plt.title('In Sample vs Benchmark')
        plt.savefig('experiment1_in_sample.png')
    else:
        plt.title('Out of Sample vs Benchmark')
        plt.savefig('experiment1_out_of_sample.png')
    plt.close()


if __name__ == "__main__":
    experiment1()