import numpy as np


class DTLearner(object):
    def __init__(self,leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        return "fyuen3"

    def find_best_feature_i(self, data):
        best_i = 0
        best_corr = -1
        for i in range(data.shape[1] - 1):
            col = data[:, i]
            corr = abs(np.corrcoef(col, data[:, -1])[0, 1])
            if (corr > best_corr):
                best_corr = corr
                best_i = i
        return best_i

    def build_tree(self, data):
        if data.shape[0] <= self.leaf_size:
            leaf = np.array([[-1, np.mean(data[:,-1]), -1, -1]])
            return leaf

        if np.all(data[:,-1] == data[0,-1]):
            return np.array([[-1, data[0][-1], -1, -1]])

        else:
            i = self.find_best_feature_i(data)
            splitVal = np.median(data[:, i], axis=0)
            # if median is the largest value of the data set, no need to split again
            if (splitVal == np.max(data[:, i], axis=0)):
                index_of_max_value = np.argmax(data[:, i])
                return np.array([[-1, data[index_of_max_value][-1], -1, -1]])

            lefttree = self.build_tree(data[data[:, i] <= splitVal])
            righttree = self.build_tree(data[data[:, i] > splitVal])
            root = np.array([[i, splitVal, 1, lefttree.shape[0]+1]])
            return np.vstack((root, lefttree, righttree))

    def add_evidence(self, data_x, data_y):
        new_data = np.ones([data_x.shape[0], data_x.shape[1] + 1])
        new_data[:, 0: data_x.shape[1]] = data_x
        new_data[:, -1] = data_y
        self.tree = self.build_tree(new_data)

    def query(self, points):
        Ypred = np.array([])
        for point in points:
            node_index = 0
            while True:
                factor = int(self.tree[node_index, 0])
                if factor == -1:
                    Ypred = np.append(Ypred,self.tree[node_index, 1])
                    break
                else:
                    split_value = self.tree[node_index, 1]
                    if point[factor] <= split_value:
                        node_index = node_index + int(self.tree[node_index, 2])
                    else:
                        node_index = node_index + int(self.tree[node_index, -1])
        return Ypred