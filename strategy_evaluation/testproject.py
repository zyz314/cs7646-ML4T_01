import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import ManualStrategy as ms
import marketsimcode as msc

import StrategyLearner as sl
from util import get_data

import experiment1 as exp1

def author():
    return 'fyuen3'

if __name__ == "__main__":
    exp1.experiment1()