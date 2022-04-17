import numpy as np
from scipy import stats

class RTLearner(object):
    def __init__(self,leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "fyuen3"

    def build_tree(self, data):
        tree = np.array([])
        flag = 0
        if (data.shape[0] <= self.leaf_size):
            tree = np.array([['leaf', data[0][-1], '-1', '-1']])
            return tree

        i= int(np.random.randint(data.shape[1] - 1))

        # if values of Xattribute are the same
        if (np.all(data[:, i] == data[0][i])):
            return np.array([['leaf', np.mean(data[:, -1]), '-1', '-1']])

        data = data[np.argsort(data[:, i])]
        splitVal = np.median(data[0:, i])
        if max(data[:, i]) == splitVal:
            return np.array([['leaf', np.mean(data[:, -1]), '-1', '-1']])

        leftTree = self.build_tree(data[data[:, i] <= splitVal])
        rightTree = self.build_tree(data[data[:, i] > splitVal])
        root = [i, splitVal, 1, leftTree.shape[0] + 1]
        tree = np.vstack((root, leftTree, rightTree))
        return tree
    def add_evidence(self, data_x, data_y):
        new_data = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        new_data[:, 0: data_x.shape[1]] = data_x
        new_data[:, -1] = data_y
        self.tree = self.build_tree(new_data)

    def query(self, points):
        row = 0
        predY = np.array([])
        for data in points:
            while (self.tree[row][0] != 'leaf'):
                X_attr = self.tree[row][0]
                X_attr = int(float(X_attr))
                if (float(data[X_attr]) <= float(self.tree[row][1])):
                    row = row + int(float(self.tree[row][2]))
                else:
                    row = row + int(float(self.tree[row][3]))
                row = int(float(row))
            if (self.tree[row][0] == 'leaf'):
                predY = np.append(predY, float(self.tree[row][1]))
                row = 0
        return predY