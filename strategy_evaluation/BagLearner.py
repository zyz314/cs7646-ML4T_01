import numpy as np


class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.learners = []
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return "fyuen3"

    def add_evidence(self, data_x, data_y):  		  	   		  	  			  		 			     			  	 
        # build learners
        rows = data_x.shape[0]
        for i in range(0, self.bags):
            choice = np.random.choice(rows, rows,replace=True)
            bag_data_x = data_x[choice]
            bag_data_y = data_y[choice]
            self.learners[i].add_evidence(bag_data_x, bag_data_y)

    def query(self, points):
        Ypred = []
        for learner in self.learners:
            Ypred.append(learner.query(points))
        return np.mean(Ypred, axis=0)