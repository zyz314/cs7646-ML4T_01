import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import ManualStrategy as ms
import marketsimcode as msc

import StrategyLearner as sl
from util import get_data

import experiment1 as exp1
import experiment2 as exp2

def author():
    return 'fyuen3'

if __name__ == "__main__":
    ms.plot_in_sample()
    ms.plot_out_of_sample()
    exp1.experiment1()
    exp2.experiment2()