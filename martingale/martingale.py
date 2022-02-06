""""""  		  	   		  	  			  		 			     			  	 
"""Assess a betting strategy.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
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
  		  	   		  	  			  		 			     			  	 
import numpy as np  		  	   		  	  			  		 			     			  	 
import matplotlib.pyplot as plt

def author():
    """  		  	   		  	  			  		 			     			  	 
    :return: The GT username of the student  		  	   		  	  			  		 			     			  	 
    :rtype: str  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    return "fyuen3"  # replace tb34 with your Georgia Tech username.
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def gtid():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    :return: The GT ID of the student  		  	   		  	  			  		 			     			  	 
    :rtype: int  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    return 903641501  # replace with your GT ID number
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def get_spin_result(win_prob):  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
    :param win_prob: The probability of winning  		  	   		  	  			  		 			     			  	 
    :type win_prob: float  		  	   		  	  			  		 			     			  	 
    :return: The result of the spin.  		  	   		  	  			  		 			     			  	 
    :rtype: bool  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    result = False  		  	   		  	  			  		 			     			  	 
    if np.random.random() <= win_prob:  		  	   		  	  			  		 			     			  	 
        result = True  		  	   		  	  			  		 			     			  	 
    return result  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
  		  	   		  	  			  		 			     			  	 
def test_code():  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    Method to test your code  		  	   		  	  			  		 			     			  	 
    """  		  	   		  	  			  		 			     			  	 
    np.random.seed(gtid())  # do this only once
    # add your code here to implement the experiments
    experiment1()
    experiment2()

# martingale strategy
def martingale_strategy(bankroll=None):
    episode_winnings = 0
    count_bet = 0
    results = np.zeros(1000)
    while (episode_winnings < 80 and count_bet<1000):
        won = False
        bet_amount = 1
        while not won:
            win_prob = 9 / 19  # set appropriately to the probability of a win
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2
                if bankroll != None and episode_winnings - bet_amount < -bankroll:
                    bet_amount = bankroll + episode_winnings

            results[count_bet] = episode_winnings
            if episode_winnings >= 80:
                results[count_bet:] = 80
                return results
            if bankroll != None:
                if episode_winnings <= -bankroll:
                    results[count_bet:] = -bankroll
                    return results

            count_bet+=1
    return results
#experient 1
def experiment1():
    #Figure 1
    episodes = 10
    plt.figure(0)
    plt.xlabel('Number of successive bet(s)')
    plt.ylabel('Winnings')
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.title('Figure 1 - 10 episodes with unlimited bankroll')
    figure1Result = np.zeros([episodes, 1000])
    for i in range(episodes):
        figure1Result[i] = martingale_strategy()
        plt.plot(range(300), figure1Result[i, 0:300])
    plt.savefig('images/Figure1.png')
    #Figure 2
    episodes = 1000
    plt.figure(1)
    plt.xlabel('Number of successive bet(s)')
    plt.ylabel('Winnings')
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.title('Figure 2 - Mean winnings with unlimited bankroll +/-stdev')
    figure2Result = np.zeros([episodes, 1000])
    for i in range(episodes):
        figure2Result[i] = martingale_strategy()
    figure2ResultMean = np.mean(figure2Result,axis=0)
    figure2ResultStd = np.std(figure2Result,axis=0)
    plt.plot(range(300), figure2ResultMean[0:300],label='Mean')
    plt.plot(range(300), figure2ResultMean[0:300]+figure2ResultStd[0:300],label='Mean + stdev')
    plt.plot(range(300), figure2ResultMean[0:300]-figure2ResultStd[0:300],label='Mean - stdev')
    plt.legend()
    plt.savefig('images/Figure2.png')
    #Figure 3
    plt.figure(2)
    plt.xlabel('Number of successive bet(s)')
    plt.ylabel('Winnings')
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.title('Figure 3 - Median winnings with unlimited bankroll +/-stdev')
    figure3ResultMedian = np.median(figure2Result, axis=0)

    plt.plot(range(300), figure3ResultMedian[0:300],label='Median')
    plt.plot(range(300), figure3ResultMedian[0:300] + figure2ResultStd[0:300],label='Median + stdev')
    plt.plot(range(300), figure3ResultMedian[0:300] - figure2ResultStd[0:300],label='Median - stdev')
    plt.legend()
    plt.savefig('images/Figure3.png')

def experiment2():
    #Figure 4
    episodes = 1000
    plt.figure(3)
    plt.xlabel('Number of successive bet(s)')
    plt.ylabel('Winnings')
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.title('Figure 4 - Mean winnings with limited bankroll +/-stdev')
    result = np.zeros([episodes, 1000])
    for i in range(episodes):
        result[i] = martingale_strategy(256)
    # np.savetxt("result.csv", result, delimiter=",")
    figure4ResultMean = np.mean(result, axis=0)
    figure4ResultStd = np.std(result, axis=0)
    plt.plot(range(300), figure4ResultMean[0:300], label='Mean')
    plt.plot(range(300), figure4ResultMean[0:300] + figure4ResultStd[0:300], label='Mean + stdev')
    plt.plot(range(300), figure4ResultMean[0:300] - figure4ResultStd[0:300], label='Mean - stdev')
    plt.legend()
    plt.savefig('images/Figure4.png')
    #Figure 5
    plt.figure(4)
    plt.xlabel('Number of successive bet(s)')
    plt.ylabel('Winnings')
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.title('Figure 5 - Median winnings with limited bankroll +/-stdev')
    figure5ResultMedian = np.median(result, axis=0)

    plt.plot(range(300), figure5ResultMedian[0:300],label='Median')
    plt.plot(range(300), figure5ResultMedian[0:300] + figure4ResultStd[0:300],label='Median + stdev')
    plt.plot(range(300), figure5ResultMedian[0:300] - figure4ResultStd[0:300],label='Median - stdev')
    plt.legend()
    plt.savefig('images/Figure5.png')

if __name__ == "__main__":
    test_code()  		  	   		  	  			  		 			     			  	 
