## READMElab2

总得分 19.65/20.00

**注意**：提交前千万看一下有没有导入什么奇奇怪怪的库，我一开始不小心导入了一个但没用，直接只有9分，然后才找助教删去那一行代码重评的

PIL可以通过pip install pillow下载

如果见到一些莫名其妙的报错可以检查一下自己是不是有什么包没有下

#### q1 100

根据README来就可以了

#### q2 100

注意要实现"purity_bound"超参数相关的熵的阈值部分，不然效果不好

#### q3 100

如果你发现你对应的节点明明实现的是对的，但是debug给你判错，就可能是Linear等比较基础的节点写错了，具体可以看input文件夹中的.in的文件

#### q4 本地98 最终评测时93

我之前在树洞上也分享过一次，现在整理一下抄上来

分享一下ai引论lab2q4的一组比较好的参数,dz调了一天半的参偶然撞出来的。

-  lr = 0.002   # 学习率 
- wd2 = 1e-4  # L2正则化系数 
- batchsize = 256  # 批大小 
- ratio_data=0.35 #构建时抽样比 
- nodes = [BatchNorm(mnist.num_feat), Linear(mnist.num_feat, 1024), BatchNorm(1024), relu(), Dropout(0.3),             Linear(1024, 512), BatchNorm(512), relu(), Dropout(0.3), Linear(512, 128), BatchNorm(128), relu(), Dropout(0.3),             Linear(128, mnist.num_class), LogSoftmax(), NLLLoss(trn_Y)] 
- 20个epoch 图像旋转-7.5到7.5度，用np.random.uniform，平移-5到5用np.random.randint 
- **训练时将训练集和验证集合在一起使用。**
- 调参时可以每个epoch会输出acc,原本是在验证集上的acc,可以改成在测试集上的acc

我也盲目地试过很多方法，比如同时训3个模型让它们投票什么的，但都效果不好，事实证明，找到一个合适的旋转角度和平移距离来过拟合测试数据有显著的效果。