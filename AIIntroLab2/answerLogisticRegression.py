import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 2  # 学习率
wd = 2e-2  # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    return X@weight+bias
    raise NotImplementedError

def sigmoid(x):
    x = np.clip(x, -700, 700)
    return np.where(x>=0,1/(1+np.exp(-x)),np.exp(x)/(1+np.exp(x)))


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    haty = predict(X, weight, bias)
    y_haty = haty * Y
    n = X.shape[0]

    # 计算损失（不包括正则化项，仅用于监控）
    loss = np.mean(np.logaddexp(0, -y_haty))

    sigma = sigmoid(y_haty)
    # 修正梯度计算，使用1 - sigma
    grad_weight = (X.T @ (-Y * (1 - sigma))) / n
    grad_bias = np.mean(-Y * (1 - sigma))

    # 应用L2正则化项的梯度
    weight = weight - lr * (grad_weight + wd * weight)
    bias = bias - lr * grad_bias

    return haty, loss, weight, bias
    raise NotImplementedError
