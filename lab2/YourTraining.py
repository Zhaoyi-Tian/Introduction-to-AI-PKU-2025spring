
from autograd.BaseNode import *
from autograd.BaseGraph import Graph
import mnist
import numpy as np
import pickle
from autograd.utils import PermIterator
from util import setseed
from scipy import ndimage

lr = 0.002   # 学习率
wd1 = 1e-5   # L1正则化系数
wd2 = 1e-4  # L2正则化系数
batchsize = 256  # 批大小
ratio_data=0.35 #构建时抽样比

setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/mtr.npy"

val_X = np.load('MNIST/valid_data.npy')
val_Y = np.load('MNIST/valid_targets.npy')
val_num_sample = val_X.shape[0]

trn_X = np.load('MNIST/train_data.npy').astype(np.float64)
trn_Y = np.load('MNIST/train_targets.npy')

trn_num_sample=mnist.trn_num_sample

trn_X=np.concatenate((trn_X,val_X),axis=0)
trn_Y=np.concatenate((trn_Y,val_Y),axis=0)
trn_num_sample=trn_num_sample+val_num_sample

num_feat = mnist.num_feat
num_class = mnist.num_class

if __name__ == "__main__":
    nodes = [BatchNorm(mnist.num_feat),
             Linear(mnist.num_feat, 1024), BatchNorm(1024), relu(), Dropout(0.3),
             Linear(1024, 512), BatchNorm(512), relu(), Dropout(0.3),
             Linear(512, 128), BatchNorm(128), relu(), Dropout(0.3),
             Linear(128, mnist.num_class), LogSoftmax(), NLLLoss(trn_Y)]
    graph = Graph(nodes)
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(trn_X.shape[0], batchsize)
    for i in range(1, 20 + 1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = trn_X[perm].copy()
            angle = np.random.uniform(7.5, -7.5, size=tX.shape[0])
            shift_val = np.random.randint(-5, 6, size=(tX.shape[0], 2))
            for j in range(tX.shape[0]):
                aft = ndimage.rotate(tX[j], angle[j], reshape=False, order=1)
                aft = ndimage.shift(aft, shift_val[j], order=1)
                tX[j] = aft
            tX = tX.reshape(tX.shape[0], -1)
            tY = trn_Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys) == np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
    with open(save_path, "wb") as f:
        pickle.dump(graph, f)