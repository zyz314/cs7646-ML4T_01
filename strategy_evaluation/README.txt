List of file submitted:
1. RTLearner.py
2. BagLearner.py
3. ManualStrategy.py
4. StrategyLearner.py
5. indicators.py
6. experiment1.py
7. experiment2.py
8. marketsimcode.py
9. testproject.py
10. README.txt
------------------------------------------------------------------------------------------------------------------
To run the code and generate all graphs and data:
PYTHONPATH=../:. python testproject.py

To run experiment1.py 
PYTHONPATH=../:. python experiment1.py

To run experiment2.py
PYTHONPATH=../:. python experiment2.py

------------------------------------------------------------------------------------------------------------------
Description of files:

RTLearner.py -  Random Tree learner being used to import into BagLearner.py to train a model
                
BagLearner.py - an ensemble learner being used to train 20 models and called by the Strategy Learner

ManualStrategy.py - contains 3 indicators : SMA, BBP, Momentum and testPolicy() to generate trade data frame
		   			contains benchmark() to generate benchmark trade data frame

StrategyLearner.py - implements BagLearner. It contains add_evidence() to train a model and testPolicy() to predict Y values

indicators.py - defines the calculation of SMA, BBP, Momentum, MACD, PPO

experiment1.py - generates portfolio values for manual strategy, strategy learner, and benchmark strategy for graph plotting

experiment2.py - compares the performance of the strategy learner for different impact values

marketsimcode.py - contains compute_portvals() converts trade data frame to portfolio value for plotting graph

testproject.py - generates all graphs and metrics for report writing
