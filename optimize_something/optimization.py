""""""  		  	   		  	  			  		 			     			  	 
"""MC1-P2: Optimize a portfolio.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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
  		  	   		  	  			  		 			     			  	 
Student Name: Fung Yi Yuen (replace with your name)  		  	   		  	  			  		 			     			  	 
GT User ID: fyuen3 (replace with your User ID)  		  	   		  	  			  		 			     			  	 
GT ID: 903641501 (replace with your GT ID)  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import datetime as dt  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import matplotlib.pyplot as plt  		  	   		  	  			  		 			     			  	 
import pandas as pd  		  	   		  	  			  		 			     			  	 
from util import get_data, plot_data
from scipy import optimize as spo
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def compute_daily_returns(df):
	"""Compute and return the daily return values."""
	daily_returns = (df / df.shift(1)) - 1
	daily_returns = daily_returns[1:]
	return daily_returns

def negative_sharpe_ratio(allocs, prices, rfr=0, sf=252):
    normed = prices / prices.iloc[0, :]
    alloced = normed * allocs
    port_val = alloced.sum(axis=1)
    daily_returns = compute_daily_returns(port_val)
    return -1* compute_sharpe_ratio(daily_returns, rfr, sf)

def compute_sharpe_ratio(daily_returns, rfr = 0,sf=252):
    return np.sqrt(sf) * (daily_returns - rfr).mean() / daily_returns.std()

def find_min_negative_sharpe_ratio_alloc(prices, sharpe_func):
    column_size = len(prices.columns)
    allocs_guess = np.ones(column_size)/column_size
    boundary = ((0,1),)*column_size
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    allocs = spo.minimize(sharpe_func, allocs_guess, args=(prices,), method='SLSQP', bounds=boundary, constraints=constraints)
    return allocs.x
# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality  		  	   		  	  			  		 			     			  	 
def optimize_portfolio(  		  	   		  	  			  		 			     			  	 
    sd=dt.datetime(2008, 1, 1),  		  	   		  	  			  		 			     			  	 
    ed=dt.datetime(2009, 1, 1),  		  	   		  	  			  		 			     			  	 
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		  	  			  		 			     			  	 
    gen_plot=False,  		  	   		  	  			  		 			     			  	 
):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  	  			  		 			     			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  	  			  		 			     			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  	  			  		 			     			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  	  			  		 			     			  	 
    statistics.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			     			  	 
    :type sd: datetime  		  	   		  	  			  		 			     			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			     			  	 
    :type ed: datetime  		  	   		  	  			  		 			     			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  	  			  		 			     			  	 
        symbol in the data directory)  		  	   		  	  			  		 			     			  	 
    :type syms: list  		  	   		  	  			  		 			     			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  	  			  		 			     			  	 
        code with gen_plot = False.  		  	   		  	  			  		 			     			  	 
    :type gen_plot: bool  		  	   		  	  			  		 			     			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  	  			  		 			     			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		  	  			  		 			     			  	 
    :rtype: tuple  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # Read in adjusted closing prices for given symbols, date range  		  	   		  	  			  		 			     			  	 
    dates = pd.date_range(sd, ed)  		  	   		  	  			  		 			     			  	 
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all.fillna(method="ffill", inplace=True)
    prices_all.fillna(method="bfill", inplace=True)

    prices = prices_all[syms]  # only portfolio symbols  		  	   		  	  			  		 			     			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		  	  			  		 			     			  	 
    normed_SPY = prices_SPY / prices_SPY.iloc[0]

      # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case  		  	   		  	  			  		 			     			  	 
    allocs = find_min_negative_sharpe_ratio_alloc(prices, negative_sharpe_ratio)
    # add code here to find the allocations

    # Get daily portfolio value
    port_val = prices_SPY  # add code here to compute daily portfolio values
    normed = prices / prices.iloc[0, :]
    alloced = normed * allocs
    port_val = alloced.sum(axis=1)

    daily_returns = compute_daily_returns(port_val)
    cr, adr, sddr, sr = [
        (port_val[-1] / port_val[0] - 1),
        daily_returns.mean(),
        daily_returns.std(),
        compute_sharpe_ratio(daily_returns),
    ]  # add code here to compute stats

  		  	   		  	  			  		 			     			  	 
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		  	  			  		 			     			  	 
    if gen_plot:  		  	   		  	  			  		 			     			  	 
        # add code to plot here  		  	   		  	  			  		 			     			  	 
        df_temp = pd.concat(  		  	   		  	  			  		 			     			  	 
            [port_val, normed_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        df_temp.plot()
        plt.grid()
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Daily Portfolio Value and SPY")
        plt.legend()
        plt.savefig('images/graph.png')
        pass  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    return allocs, cr, adr, sddr, sr  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def test_code():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    This function WILL NOT be called by the auto grader.  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    start_date = dt.datetime(2009, 1, 1)  		  	   		  	  			  		 			     			  	 
    end_date = dt.datetime(2010, 1, 1)  		  	   		  	  			  		 			     			  	 
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # Assess the portfolio  		  	   		  	  			  		 			     			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		  	  			  		 			     			  	 
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # Print statistics  		  	   		  	  			  		 			     			  	 
    print(f"Start Date: {start_date}")  		  	   		  	  			  		 			     			  	 
    print(f"End Date: {end_date}")  		  	   		  	  			  		 			     			  	 
    print(f"Symbols: {symbols}")  		  	   		  	  			  		 			     			  	 
    print(f"Allocations:{allocations}")  		  	   		  	  			  		 			     			  	 
    print(f"Sharpe Ratio: {sr}")  		  	   		  	  			  		 			     			  	 
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  	  			  		 			     			  	 
    print(f"Average Daily Return: {adr}")  		  	   		  	  			  		 			     			  	 
    print(f"Cumulative Return: {cr}")  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    # This code WILL NOT be called by the auto grader  		  	   		  	  			  		 			     			  	 
    # Do not assume that it will be called  		  	   		  	  			  		 			     			  	 
    test_code()  		  	   		  	  			  		 			     			  	 
