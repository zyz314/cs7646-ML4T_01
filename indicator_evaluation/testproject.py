import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from util import get_data

import TheoreticallyOptimalStrategy as tos
import indicators as ind


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "fyuen3"


if __name__ == "__main__":
    df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades_benchmark = tos.benchmark(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                        shares=1000)
    tos.plot_graph(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), df_trades=df_trades,
                   df_trades_benchmark=df_trades_benchmark)
    ind.plotGraph(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31))