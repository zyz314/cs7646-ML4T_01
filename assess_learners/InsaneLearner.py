import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = []
        self.verbose = verbose
        for i in range(20):
            self.learners.append(bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=self.verbose))
    def author(self):
        return "fyuen3"
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)
    def query(self, points):
        Ypred = []
        for learner in self.learners:
            Ypred.append(learner.query(points))
        return np.mean(Ypred, axis=0)