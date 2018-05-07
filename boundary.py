import numpy as np
from scipy import spatial


class Node:
    def __init__(self, node_index, t, l, x):
        self.node_index = node_index
        self.child_nodes = []
        self.T = np.array([t])
        self.L = np.array([l])
        self.X = np.array([x])

    def add(self, node_index, t, l, x):
        self.child_nodes.append(Node(node_index, t, l, x))
        self.T = np.vstack([self.T, t])
        self.L = np.vstack([self.L, l])
        self.X = np.vstack([self.X, x])

    @property
    def label(self): return self.L[0]
    @property
    def num_children(self): return len(self.child_nodes)


pdists__ = lambda arr, vec: spatial.distance.cdist(arr, np.array([vec]), 'euclidean')

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1,keepdims=1)
    #print softmax_2(np.array([[1,2,3],[0,0,0],[.1,10,-199],[10,10,10]]))

    
class Tree:
    def __init__(self):
        self.root = None
        self.size = 0

    def train(self, t, l, x):
        if self.root is None:
            self.root = Node(0, t, l, x)
            self.size = 1
            return True
        else:
            v = self.query(t)
            if np.argmax(v.label) == np.argmax(l):
                return False
            else:
                v.add(self.size, t, l, x)
                self.size += 1
                return True

    def query__(self, v, t):
        return np.argmin(pdists__(v.T, t)) - 1

    def query(self, t):
        v = self.root
        while 1:
            if v.num_children == 0:
                return v
            else:
                ind = self.query__(v, t)
                if ind<0:
                    return v
                else:
                    v = v.child_nodes[ind]

    def query_parent(self, t):
        v = self.root
        p = v
        while 1:
            if v.num_children == 0:
                return p
            else:
                ind = self.query__(v, t)
                if ind<0:
                    return p
                else:
                    p = v
                    v = v.child_nodes[ind]

    def query_neighbors(self, t):
        p = self.query_parent(t)
        return p.X, p.L
    
    def query_neighbor_inds(self, t):
        p = self.query_parent(t)
        inds = [child.node_index for child in p.child_nodes]
        inds.append(p.node_index)
        return np.array(inds)

    def infer_probs(self, t, sigma):
        p = self.query_parent(t)
        dists = pdists__(p.T, t).T
        smax = softmax(-dists/sigma)
        return np.matmul(smax, p.L)[0]
    
def build_boundary_tree_ex(data, labels, meta):
    result = []
    b_tree = Tree()
    for t, l, x in zip(data, labels, meta):
        result.append(b_tree.train(t, l, x))
    return b_tree, result

    
def build_boundary_tree(data, labels, meta):
    return build_boundary_tree_ex(data, labels, meta)[0]


class Forest:
    def __init__(self, dim, n, k):
        self.dim = dim
        self.n = n
        self.k = k
        self.trees = []

    def train(self, y, l):
        if len(self.trees) < self.n:
            self.trees.append(Tree(self.dim, self.k))

        for T in self.trees:
            T.train(y, l)


__dim = lambda arg: len(arg[0])


class Set:
    def __init__(self, dim_x, dim_y):
        self.values = np.zeros(shape=(0, dim_x))
        self.labels = np.zeros(shape=(0, dim_y))

    def __add(self, y, l):
        self.values = np.vstack([self.values, y])
        self.labels = np.vstack([self.labels, l])

    @property
    def size(self): return len(self.labels)

    def train(self, y, l):
        if self.values.size == 0:
            self.__add(y, l)
            return True
        else:
            value, label = self.query(y)
            if np.argmax(label) == np.argmax(l):
                return False
            else:
                self.__add(y, l)
                return True

    def query(self, y):
        dists = spatial.distance.cdist(
            self.values, np.array([y]), 'euclidean')
        ind = np.argmin(dists)
        value = self.values[ind]
        label = self.labels[ind]
        return value, label


def build_boundary_set_ex(data, labels):
    result = []
    b_set = Set(len(data[0]), len(labels[0]))
    for y, l in zip(data, labels):
        result.append(b_set.train(y, l))
    return b_set, result


def build_boundary_set(data, labels):
    return build_boundary_set_ex(data, labels)[0]


def __draw(p, plt):
    if p.child_nodes:
        for v in p.child_nodes:
            plt.plot((v.value[0], p.value[0]),
                     (v.value[1], p.value[1]), 'g-')
            __draw(v, plt)


def simulate_tree(k, data, labels, plt):
    t = Tree(__dim(data), k)
    for y, l in zip(data, labels):
        t.train(y, l)

    plt.scatter(data[:, 0], data[:, 1], marker='.', s=20, c=labels)
    __draw(t.root, plt)
    plt.axis('equal')


def simulate_set(data, labels, plt):
    s = Set(__dim(data))
    for y, l in zip(data, labels):
        s.train(y, l)

    plt.scatter(data[:, 0], data[:, 1], marker='.', s=5, edgecolor='none', c=labels)
    print(len(s.labels), len(s.values))
    plt.scatter(s.values[:, 0], s.values[:, 1], marker='s', s=30, c=s.labels, edgecolor='1')
    plt.axis('equal')


def simulate_forest(n, k, data, labels, plt):
    f = Forest(__dim(data), n, k)
    s = Set(len(data[0]))
    for y, l in zip(data, labels):
        f.train(y, l)
        s.train(y, l)

    for T, i in zip(f.trees, range(10)):
        plt.subplot(1, 3, i+1)
        plt.scatter(data[:, 0], data[:, 1], marker='.', s=20, c=labels)
        plt.scatter(s.values[:, 0], s.values[:, 1], marker='s', s=70,
                    c=s.labels, edgecolor='1')
        __draw(T.root, plt)
        plt.axis('equal')


def run_tests():
    from sklearn.datasets import make_moons, make_classification
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    #from loading import make_data

    np.random.seed(676)
    #data, labels = make_data(n_samples=10000)

    data, labels = make_moons(n_samples=10000, shuffle=True, noise=None, random_state=None) # make_circles

    #data, labels = make_moons(n_samples=1000, shuffle=True, noise=None, random_state=None)
    #data, labels = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    #simulate_forest(3, 3, data, labels, plt)
    #simulate_tree(3, data, labels, plt)

    num = 10
    start = time.time()
    print("starting")
    for _ in range(num):
        s = Set(len(data[0]))
        for y, l in zip(data, labels):
            s.train(y, l)
        #print(s.labels)
    end = time.time()
    print((end - start)/num)

    simulate_set(data, labels, plt)
    plt.show()


if __name__ == "__main__":
    run_tests()
