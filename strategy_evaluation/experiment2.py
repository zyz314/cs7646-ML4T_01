import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import marketsimcode as msc

import StrategyLearner as sl
from util import get_data

def author():
    return 'fyuen3'

def experiment2():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = "JPM"

    learner = sl.StrategyLearner(verbose=False, impact=0.0)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
    df_trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    df_orders = df_trades.copy()
    df_orders['Symbol'] = symbol
    df_orders['Order'] = np.where(df_orders['Trades'] > 0, 'BUY', 'SELL')
    df_orders['Shares'] = np.abs(df_orders['Trades'])

    port_vals = msc.compute_portvals(df_orders, start_val=100000, commission=0.0, impact=0.0)
    port_vals_norm = port_vals / port_vals.iloc[0]

    learner1 = sl.StrategyLearner(verbose=False, impact=0.002)
    learner1.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
    df_trades1 = learner1.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    df_orders1 = df_trades1.copy()
    df_orders1['Symbol'] = symbol
    df_orders1['Order'] = np.where(df_orders1['Trades'] > 0, 'BUY', 'SELL')
    df_orders1['Shares'] = np.abs(df_orders1['Trades'])
    port_vals1 = msc.compute_portvals(df_orders1, start_val=100000, commission=0.0, impact=0.002)
    port_vals_norm1 = port_vals1 / port_vals1.iloc[0]

    learner2 = sl.StrategyLearner(verbose=False, impact=0.05)
    learner2.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
    df_trades2 = learner2.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    df_orders2 = df_trades2.copy()
    df_orders2['Symbol'] = symbol
    df_orders2['Order'] = np.where(df_orders2['Trades'] > 0, 'BUY', 'SELL')
    df_orders2['Shares'] = np.abs(df_orders2['Trades'])
    port_vals2 = msc.compute_portvals(df_orders2, start_val=100000, commission=0.0, impact=0.05)
    port_vals_norm2 = port_vals2 / port_vals2.iloc[0]

    learner3 = sl.StrategyLearner(verbose=False, impact=0.1)
    learner3.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
    df_trades3 = learner3.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    df_orders3 = df_trades3.copy()
    df_orders3['Symbol'] = symbol
    df_orders3['Order'] = np.where(df_orders3['Trades'] > 0, 'BUY', 'SELL')
    df_orders3['Shares'] = np.abs(df_orders3['Trades'])
    port_vals3 = msc.compute_portvals(df_orders3, start_val=100000, commission=0.0, impact=0.1)
    port_vals_norm3 = port_vals3 / port_vals3.iloc[0]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax = port_vals_norm.plot(label='Impact 0.0')
    port_vals_norm1.plot(ax=ax, label='Impact 0.002')
    port_vals_norm2.plot(ax=ax, label='Impact 0.05')
    port_vals_norm3.plot(ax=ax, label='Impact 0.1')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.title("Strategy Learner with different impact values")
    plt.savefig("Experiment2.png")

    port_cr, port_adr, port_stddr, port_sr = msc.assess_portfolio(port_vals)
    port1_cr, port1_adr, port1_stddr, port1_sr = msc.assess_portfolio(port_vals1)
    port2_cr, port2_adr, port2_stddr, port2_sr = msc.assess_portfolio(port_vals2)
    port3_cr, port3_adr, port3_stddr, port3_sr = msc.assess_portfolio(port_vals3)

    print(f"Cumulative return of the impact 0: {port_cr}")
    print(f"Stdev of daily returns of impact 0: {port_stddr}")
    print(f"Average daily returns of impact 0: {port_adr}")
    print(f"Sharpe ratio of impact 0: {port_sr}")

    print(f"Cumulative return of the impact 0.002: {port1_cr}")
    print(f"Stdev of daily returns of impact 0.002: {port1_stddr}")
    print(f"Average daily returns of impact 0.002: {port1_adr}")
    print(f"Sharpe ratio of impact 0.002: {port1_sr}")

    print(f"Cumulative return of the impact 0.05: {port2_cr}")
    print(f"Stdev of daily returns of impact 0.05: {port2_stddr}")
    print(f"Average daily returns of impact 0.05: {port2_adr}")
    print(f"Sharpe ratio of impact 0.05: {port2_sr}")

    print(f"Cumulative return of the impact 0.1: {port3_cr}")
    print(f"Stdev of daily returns of impact 0.1: {port3_stddr}")
    print(f"Average daily returns of impact 0.1: {port3_adr}")
    print(f"Sharpe ratio of impact 0.1: {port3_sr}")