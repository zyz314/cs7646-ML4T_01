""""""  		  	   		  	  			  		 			     			  	 
"""  		  	   		  	  			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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
"""  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import math  		  	   		  	  			  		 			     			  	 
import sys  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
import numpy as np
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl

import LinRegLearner as lrl
import matplotlib.pyplot as plt
  		  	   		  	  			  		 			     			  	 
if __name__ == "__main__":  		  	   		  	  			  		 			     			  	 
    if len(sys.argv) != 2:  		  	   		  	  			  		 			     			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		  	  			  		 			     			  	 
        sys.exit(1)  		  	   		  	  			  		 			     			  	 
    inf = open(sys.argv[1])
    if sys.argv[1] == "Data/Istanbul.csv":
        inf = open(sys.argv[1])
        data = np.genfromtxt(inf, delimiter=',')
        data = data[1:, 1:]
    else:
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )
  		  	   		  	  			  		 			     			  	 
    # compute how much of the data is training and testing  		  	   		  	  			  		 			     			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		  	  			  		 			     			  	 
    test_rows = data.shape[0] - train_rows  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # separate out training and testing data  		  	   		  	  			  		 			     			  	 
    train_x = data[:train_rows, 0:-1]  		  	   		  	  			  		 			     			  	 
    train_y = data[:train_rows, -1]  		  	   		  	  			  		 			     			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		  	  			  		 			     			  	 
    test_y = data[train_rows:, -1]  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    print(f"{test_x.shape}")  		  	   		  	  			  		 			     			  	 
    print(f"{test_y.shape}")  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # create a learner and train it  		  	   		  	  			  		 			     			  	 
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner

    # learner = dt.DTLearner(leaf_size=10, verbose=False)  # constructor
    # Experiment 1
    rmse_in_sample = []
    rmse_out_of_sample = []

    for i in range(1, 70):
        learner = dt.DTLearner(leaf_size=i, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse_in_sample.append(math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0]))

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse_out_of_sample.append(math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0]))

    x = range(1, 70)
    plt.figure(1)
    plt.plot(x, rmse_in_sample, label="in-sample", linewidth=2.0)
    plt.plot(x, rmse_out_of_sample, label="out-of-sample", linewidth=2.0)
    plt.xlabel("Leaf Size")
    plt.ylabel("Root Mean Squared Errors")
    plt.legend(loc="lower right")
    plt.title("Figure 1: RMSE of Decision Tree Learner with different leaf size")
    plt.savefig('images/Exp1_fig1.png')
    plt.close()

    # Experiment 2
    rmse_in_sample = []
    rmse_out_of_sample = []

    for i in range(1, 70):
        learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":i}, bags = 20, boost = False, verbose = False)
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse_in_sample.append(math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0]))

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rmse_out_of_sample.append(math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0]))

    x = range(1, 70)
    plt.figure(2)
    plt.plot(x, rmse_in_sample, label="in-sample", linewidth=2.0)
    plt.plot(x, rmse_out_of_sample, label="out-of-sample", linewidth=2.0)
    plt.xlabel("Leaf Size")
    plt.ylabel("Root Mean Squared Errors")
    plt.legend(loc="lower right")
    plt.title("Figure 2:\nRMSE of Bagging with Decision Tree Learner with different leaf size \n(20 bags)")
    plt.savefig('images/Exp2_fig2.png')
    plt.close()

    # Experiment 3
    #MAE
    mae_dt_in_sample = []
    mae_dt_out_of_sample = []
    mae_rt_in_sample = []
    mae_rt_out_of_sample = []
    for i in range(1, 70):
        learner = dt.DTLearner(leaf_size=i, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        mae_dt_in_sample.append(np.abs(train_y - pred_y).sum() / train_y.shape[0])

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        mae_dt_out_of_sample.append(np.abs(test_y - pred_y).sum() / test_y.shape[0])

        learner = rt.RTLearner(leaf_size=i, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        mae_rt_in_sample.append(np.abs(train_y - pred_y).sum() / train_y.shape[0])

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        mae_rt_out_of_sample.append(np.abs(test_y - pred_y).sum() / test_y.shape[0])

    x = range(1, 70)
    plt.figure(3)
    plt.plot(x, mae_dt_in_sample, label="in sample error of DTLearner", linewidth=2.0)
    plt.plot(x, mae_dt_out_of_sample, label="out of sample error of DT Learner", linewidth=2.0)
    plt.plot(x, mae_rt_in_sample, label="in sample error of RTLearner", linewidth=2.0)
    plt.plot(x, mae_rt_out_of_sample, label="out of sample error of RT Learner", linewidth=2.0)
    plt.xlabel("Leaf Size")
    plt.ylabel("Mean Absolute Error")
    plt.legend(loc="lower right")
    plt.title("Figure 3: MAE of DTLearner vs RTLearner with different leaf size")
    plt.savefig('images/Exp3_fig3.png')
    plt.close()

    #MAPE
    mape_dt_in_sample = []
    mape_dt_out_of_sample = []
    mape_rt_in_sample = []
    mape_rt_out_of_sample = []
    for i in range(1, 70):
        learner = dt.DTLearner(leaf_size=i, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        mape_dt_in_sample.append(np.mean(np.abs((train_y - pred_y)/train_y))*100)

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        mape_dt_out_of_sample.append(np.mean(np.abs((test_y - pred_y)/test_y))*100)

        learner = rt.RTLearner(leaf_size=i, verbose=False)  # constructor
        learner.add_evidence(train_x, train_y)  # train it
        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        mape_rt_in_sample.append(np.mean(np.abs((train_y - pred_y)/train_y))*100)

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        mape_rt_out_of_sample.append(np.mean(np.abs((test_y - pred_y)/test_y))*100)

    x = range(1, 70)
    plt.figure(4)
    plt.plot(x, mape_dt_in_sample, label="in sample error of DTLearner", linewidth=2.0)
    plt.plot(x, mape_dt_out_of_sample, label="out of sample error of DT Learner", linewidth=2.0)
    plt.plot(x, mape_rt_in_sample, label="in sample error of RTLearner", linewidth=2.0)
    plt.plot(x, mape_rt_out_of_sample, label="out of sample error of RT Learner", linewidth=2.0)
    plt.xlabel("Leaf Size")
    plt.ylabel("Mean Absolute Percentage Error")
    plt.legend(loc="lower right")
    plt.title("Figure 4: MAPE of DTLearner vs RTLearner with different leaf size")
    plt.savefig('images/Exp3_fig4.png')
    plt.close()
  		  	   		  	  			  		 			     			  	 
    # evaluate in sample
    learner = lrl.LinRegLearner(verbose=True)
    learner.add_evidence(train_x, train_y)  # train it
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print("In sample results")  		  	   		  	  			  		 			     			  	 
    print(f"RMSE: {rmse}")  		  	   		  	  			  		 			     			  	 
    c = np.corrcoef(pred_y, y=train_y)  		  	   		  	  			  		 			     			  	 
    print(f"corr: {c[0,1]}")  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    # evaluate out of sample  		  	   		  	  			  		 			     			  	 
    pred_y = learner.query(test_x)  # get the predictions  		  	   		  	  			  		 			     			  	 
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		  	  			  		 			     			  	 
    print()  		  	   		  	  			  		 			     			  	 
    print("Out of sample results")  		  	   		  	  			  		 			     			  	 
    print(f"RMSE: {rmse}")  		  	   		  	  			  		 			     			  	 
    c = np.corrcoef(pred_y, y=test_y)  		  	   		  	  			  		 			     			  	 
    print(f"corr: {c[0,1]}")  		  	   		  	  			  		 			     			  	 
