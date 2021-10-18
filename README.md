# paddle paddle DANet复现总结： #

## 整体 ##

### 1.数据部分

使用Cityscapes-gtFine数据集，类别19类，包含背景20类

### 2.模型部分

使用Resnet 101作为backbone提取特征，以及CAM、PAM两个attention模块

### 3.Loss部分

softmax_with_cross_entropy
CAM、PAM、CAM+PAM
Weight[0.3,0.3,0.4]

### 4.Optimizer部分

MomentumOptimizer
momentum=0.9，l2_decay=0.0001
learning rate
warmup(2 epoch):由1e-4线性—>1e-1
polynomial_decay:1e-1->(1e-1-1e-4)((1-iter/total_iter)**0.9)+1e-4

在AI Studio中训练，目前只训练了93个epoch




# 对齐：

### 1.模型结构对齐：

1. 定义PyTorch模型，加载权重，固定seed，基于numpy生成随机数，转换为PyTorch可以处理的tensor，送入网络，获取输出，使用reprod_log保存结果。
2. 定义PaddlePaddle模型，加载权重，固定seed，基于numpy生成随机数，转换为PaddlePaddle可以处理的tensor，送入网络，获取输出，使用reprod_log保存结果。
3. 使用reprod_log排查diff，小于阈值，完成自测。



### 2.评估指标对齐：

miou 对齐ok



### 3.损失函数对齐：
torch.nn.CrossEntropyLoss()
paddle.nn.CrossEntropyLoss() 
对齐ok


### 4.反向初次对齐：



### 5.训练对齐：