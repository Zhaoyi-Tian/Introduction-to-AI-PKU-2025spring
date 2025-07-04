import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.001   # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 1e-4  # L2正则化
batchsize = 128

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    nodes = [StdScaler(mnist.mean_X, mnist.std_X),
             Linear(mnist.num_feat,1024),BatchNorm(1024), relu(), Dropout(0.35),
             Linear(1024, mnist.num_class), BatchNorm(mnist.num_class),Softmax(),
             CrossEntropyLoss(Y)]
    graph = Graph(nodes)
    return graph
