""""""  		  	   		  	  			  		 			     			  	 
"""MC2-P1: Market simulator.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			     			  	 
All Rights Reserved  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			     			  	 
or edited.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			     			  	 
GT honor code violation.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
Student Name: FUNG YI YUEN (replace with your name)  		  	   		  	  			  		 			     			  	 
GT User ID: fyuen3 (replace with your User ID)  		  	   		  	  			  		 			     			  	 
GT ID: 903641501 (replace with your GT ID)  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import datetime as dt  		  	   		  	  			  		 			     			  	 
import os  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import pandas as pd  		  	   		  	  			  		 			     			  	 
from util import get_data, plot_data  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def compute_portvals(  		  	   		  	  			  		 			     			  	 
    orders_df,
    start_val=1000000,  		  	   		  	  			  		 			     			  	 
    commission=9.95,  		  	   		  	  			  		 			     			  	 
    impact=0.005,  		  	   		  	  			  		 			     			  	 
):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Computes the portfolio values.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param orders_df: order dataframe  		  	   		  	  			  		 			     			  	 
    :type orders_df: pandas.DataFrame  		  	   		  	  			  		 			     			  	 
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
            price = price * (1-impact)
        else:
            price = price * (1+impact)

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

def author():
    return "fyuen3"

def test_code():
    """  		  	   		  	  			  		 			     			  	 
    Helper function to test code  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		  	  			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		  	  			  		 			     			  	 
    # Define input parameters  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    of = "./orders/orders-02.csv"
    sv = 1000000  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # Process orders  		  	   		  	  			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		  	  			  		 			     			  	 
    if isinstance(portvals, pd.DataFrame):  		  	   		  	  			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		  	  			  		 			     			  	 
    else:  		  	   		  	  			  		 			     			  	 
        "warning, code did not return a DataFrame"  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # Get portfolio stats  		  	   		  	  			  		 			     			  	 
    orders_df = pd.read_csv(of, index_col='Date', parse_dates=True, na_values=['nan'])

    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = assess_portfolio(port_val=portvals)

    spy_prices_df = get_data(['IBM'], pd.date_range(start_date, end_date))
    spy_prices_df.drop('IBM', axis=1, inplace=True)
    spy_prices_df.ffill(inplace=True)
    spy_prices_df.bfill(inplace=True)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = assess_portfolio(port_val=spy_prices_df["SPY"])
  		  	   		  	  			  		 			     			  	 
    # Compare portfolio against $SPX  		  	   		  	  			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		  	  			  		 			     			  	 
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		  	  			  		 			     			  	 
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		  	  			  		 			     			  	 
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		  	  			  		 			     			  	 
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    test_code()  		  	   		  	  			  		 			     			  	 
