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

    prices_change_df = prices_df.diff().shift(-1)

    holding_df = pd.DataFrame(index=prices_df.index, data=np.zeros(prices_df.shape), columns=prices_df.columns)
    df_trades = pd.DataFrame(index=prices_df.index, columns=['Trades'])

    for i in range(holding_df.shape[0]):
        index = holding_df.index[i]
        price_change = prices_change_df.iloc[i][symbol]
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


def benchmark(symbol, sd, ed, shares):
    prices_df = get_data([symbol], pd.date_range(sd, ed))
    df_trades = pd.DataFrame(index=prices_df.index, columns=['Trades'])
    df_trades['Trades'] = 0

    start = df_trades.index.min()
    df_trades.loc[start, 'Trades'] = shares
    return df_trades

def plot_graph(symbol,sd, ed, df_trades, df_trades_benchmark):
    df_orders = df_trades.copy()
    df_orders['Symbol'] = symbol
    df_orders['Order'] = np.where(df_orders['Trades'] > 0, 'BUY', 'SELL')
    df_orders['Shares'] = np.abs(df_orders['Trades'])

    port_vals = compute_portvals(df_orders, start_val=100000, commission=0.0, impact=0.0)
    df_orders = df_trades_benchmark.copy()
    df_orders['Symbol'] = symbol
    df_orders['Order'] = np.where(df_orders['Trades'] > 0, 'BUY', 'SELL')
    df_orders['Shares'] = np.abs(df_orders['Trades'])

    benchmark_port_vals = compute_portvals(df_orders, start_val=100000, commission=0.0, impact=0.0)
    normed_port = port_vals / port_vals.iloc[0]
    normed_bench = benchmark_port_vals / benchmark_port_vals.iloc[0]

    plt.figure(figsize=(12, 6.5))
    plt.plot(normed_port, label="Portfolio", color='red', lw=2)
    plt.plot(normed_bench, label="Benchmark", color='purple', lw=1.2)

    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.title('Theoretically Optimal Strategy (%s)' % symbol)
    plt.savefig('3_3.png')
    plt.close()

    # Print statistics
    print(f"Start Date: {sd}")
    print(f"End Date: {ed}")
    print(f"Symbol: {symbol}")

    port_cr, port_adr, port_stddr, port_sr = assess_portfolio(port_vals)
    bench_cr, bench_adr, bench_stddr, bench_sr = assess_portfolio(benchmark_port_vals)

    print(f"Cumulative return of the benchmark: {bench_cr}")
    print(f"Cumulative return of the portfolio: {port_cr}")

    print(f"Stdev of daily returns of benchmark: {bench_stddr}")
    print(f"Stdev of daily returns of portfolio: {port_stddr}")

    print(f"Mean of daily returns of benchmark: {bench_adr}")
    print(f"Mean of daily returns of portfolio: {port_adr}")

def compute_portvals(
        orders_df,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    stock_list = list(orders_df['Symbol'].unique())
    prices_df = get_data(stock_list, pd.date_range(start_date, end_date))
    prices_df.ffill(inplace=True)
    prices_df.bfill(inplace=True)
    if 'SPY' not in stock_list:
        prices_df.drop(labels='SPY', axis=1)
    prices_df['Cash'] = 1.0

    trade_df = pd.DataFrame(index=prices_df.index, data=np.zeros(prices_df.shape), columns=prices_df.columns)
    trade_df.iloc[0, -1] = start_val

    for i in range(orders_df.shape[0]):
        index = orders_df.index[i]
        symbol = orders_df.iloc[i]['Symbol']
        order = orders_df.iloc[i]['Order']
        shares = orders_df.iloc[i]['Shares']
        price = prices_df.loc[index, symbol]
        if order == 'SELL':
            shares = shares * -1
            price = price * (1 - impact)
        else:
            price = price * (1 + impact)

        trade_df.loc[index, 'Cash'] = trade_df.loc[index, 'Cash'] - commission - price * shares
        trade_df.loc[index, symbol] = trade_df.loc[index, symbol] + shares

    holding_df = trade_df.cumsum()
    holding_values_df = prices_df * holding_df

    portvals = holding_values_df.sum(axis=1)
    return portvals


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = (df / df.shift(1)) - 1
    daily_returns = daily_returns[1:]  # set daily returns for row 0 to 0
    return daily_returns


def assess_portfolio(
        port_val=np.zeros(1),
        rfr=0.0,
        sf=252.0,
):
    # Get portfolio statistics (note: std_daily_ret = volatility)
    daily_returns = compute_daily_returns(port_val)

    cr, adr, sddr, sr = [
        (port_val[-1] / port_val[0] - 1),
        daily_returns.mean(),
        daily_returns.std(),
        np.sqrt(sf) * (daily_returns - rfr).mean() / daily_returns.std(),
    ]

    return cr, adr, sddr, sr
