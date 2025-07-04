import math
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *

class NullModel:
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:
    def __init__(self):
        self.dataset = traindataset() # 完整训练集，需较长加载时间
        #self.dataset = minitraindataset() # 用来调试的小训练集，仅用于检查代码语法正确性

        # 以下内容可根据需要自行修改，不修改也可以完成本题
        self.token_num = [{}, {}] # token在正负样本中出现次数
        self.V = 0 #语料库token数量
        self.pos_neg_num = [0, 0] # 正负样本数量
        self.sum_good = 0
        self.sum_bad = 0
        self.token_p = [{}, {}]
        self.p_good = 0
        self.p_bad = 0
        self.Laplace = 1
        self.count()
    def count(self):
        # TODO: YOUR CODE HERE
        for text, label in self.dataset:
            if label==1:
                self.pos_neg_num[0] += 1
                for word in text:
                    if (word not in self.token_num[0]) and (word not in self.token_num[1]):
                        self.V += 1
                    if word in self.token_num[0]:
                        self.token_num[0][word] += 1
                    else:
                        self.token_num[0][word] = 1
                    self.sum_good +=1
            else:
                self.pos_neg_num[1] += 1
                for word in text:
                    if (word not in self.token_num[0]) and (word not in self.token_num[1]):
                        self.V += 1
                    if word in self.token_num[1]:
                        self.token_num[1][word] += 1
                    else:
                        self.token_num[1][word] = 1
                    self.sum_bad +=1
        a=self.pos_neg_num
        self.p_good= a[0] / (a[1] + a[0])
        self.p_bad = a[1] / (a[1] + a[0])
        for word,count in self.token_num[0].items():
            self.token_p[0][word]=(count+self.Laplace)/(self.sum_good+self.V)
        for word,count in self.token_num[1].items():
            self.token_p[1][word]=(count+self.Laplace)/(self.sum_bad+self.V)
        # 提示：统计token分布不需要返回值
        #raise NotImplementedError # 由于没有返回值，提交时请删掉这一行

    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 返回1或0代表当前句子分类为正/负样本
        p_good=self.p_good
        p_bad=self.p_bad

        for word in text:
            if (word not in self.token_num[0]) and (word not in self.token_num[1]):
                p_good*=self.Laplace/(self.sum_good+self.V)
                p_bad*=self.Laplace/(self.sum_bad+self.V)
            elif word not in self.token_num[0]:
                p_good *= self.Laplace / (self.sum_good + self.V)
                p_bad *= self.token_p[1][word]
            elif word not in self.token_num[1]:
                p_good *= self.token_p[0][word]
                p_bad *= self.Laplace / (self.sum_bad + self.V)
            else:
                p_good*=self.token_p[0][word]
                p_bad*=self.token_p[1][word]
        if p_good > p_bad:
            return 1
        else:
            return 0
        
        raise NotImplementedError


def buildGraph(dim, num_classes, L): #dim: 输入一维向量长度, num_classes:分类数
    # 以下类均需要在BaseNode.py中实现
    # 也可自行修改模型结构
    nodes = [Attention(dim), relu(), LayerNorm((L, dim)), ResLinear(dim), relu(), LayerNorm((L, dim)), Mean(1), Linear(dim, num_classes), LogSoftmax(), NLLLoss(num_classes)]
    
    graph = Graph(nodes)
    return graph


save_path = "model/attention.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector
        
    def __call__(self, text, max_len=50):
        # TODO: YOUR CODE HERE
        # 利用self.emb将句子映射为一个二维向量（LxD），注意，同时需要修改训练代码中的网络维度部分
        # 默认长度L为50，特征维度D为100
        # 提示: 考虑句子如何对齐长度，且可能存在空句子情况（即所有单词均不在emd表内）
        # 初始化输出矩阵（LxD），默认用全零向量填充
        output = np.zeros((max_len, 100), dtype=np.float32)
        if not text:
            return output
        valid_words=[]
        for word in text:
            if word in self.emb:
                valid_words.append(word)
        num_valid = len(valid_words)
        for i, word in enumerate(valid_words[0 : max(max_len, num_valid)]):
            output[i] = self.emb[word]
        return output
        raise NotImplementedError


class AttentionModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()

    def __call__(self, text, max_len=50):
        X = self.embedding(text, max_len)
        X = np.expand_dims(X, 0)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=-1)
        return haty[0]


class QAModel():
    def __init__(self):
        self.document_list = get_document()

    def tf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回单词在文档中的频度
        # document变量结构请参考fruit.py中get_document()函数
        d=self.document_list[document]
        tokens=d["document"]
        N=len(tokens)
        n=0
        for w in tokens:
            if word==w:
                n+=1
        return np.log10(n/N+1)
        
        raise NotImplementedError  

    def idf(self, word):
        # TODO: YOUR CODE HERE
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        # 注意python整除与整数除法的区别
        D=0
        d=0
        for document in self.document_list:
            if word in document["document"]:
                D+=1
                d+=1
            else:
                D+=1
        return np.log10(D/(1+d))


        raise NotImplementedError
    
    def tfidf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回TF-IDF值
        tf=self.tf(word, document)
        idf=self.idf(word)
        return tf*idf
        raise NotImplementedError

    def __call__(self, query):
        query = tokenize(query) # 将问题token化
        # TODO: YOUR CODE HERE
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，可以参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数
        max_res=0
        for document in range(len(self.document_list)):
            res=0
            for w in query:
                res+=self.tfidf(w, document)
            if res>max_res:
                max_res=res
                document_res=document
        doc=self.document_list[document_res]
        max_sum_idf=0
        max_bi=0
        for l in doc["sentences"]:
            sum_idf=0
            N=len(l[0])
            n=0
            for w in query:
                if w in l[0]:
                    sum_idf+=self.idf(w)
                    n+=1
            bi=n/N
            if sum_idf>max_sum_idf:
                res_line=l[1]
                max_bi=bi
                max_sum_idf=sum_idf
            elif sum_idf==max_sum_idf and bi>max_bi:
                max_sum_idf=sum_idf
                max_bi=bi
                res_line=l[1]
        return res_line



        raise NotImplementedError

modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "Attn": AttentionModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 6e-3   # 学习率
    wd1 = 1e-4  # L1正则化
    wd2 = 1e-5  # L2正则化
    batchsize = 64
    max_epoch = 3
    
    max_L = 50
    num_classes = 2
    feature_D = 100
    
    graph = buildGraph(feature_D, num_classes, max_L) # 维度可以自行修改

    # 训练
    # 完整训练集训练有点慢
    best_train_acc = 0
    dataloader = traindataset(shuffle=True) # 完整训练集
    #dataloader = minitraindataset(shuffle=True) # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        for text, label in dataloader:
            x = embedding(text, max_L)
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)
            Y.append(label)
            cnt += 1
            if cnt == batchsize:
                X = np.stack(X, 0)
                Y = np.concatenate(Y, 0)
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=-1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)