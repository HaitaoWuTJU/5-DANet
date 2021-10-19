

import paddle
import numpy as np
np.random.seed(1)
import torch
def torchRes(pre,target):
    pre=torch.from_numpy(pre).cuda()
    target=torch.from_numpy(target).cuda()
    criterion=torch.nn.CrossEntropyLoss()
    res=criterion(pre,target)
    return res.data.cpu().numpy()

def paddleRes(pre,target):
    
    pre=paddle.to_tensor(pre)
    pre=paddle.transpose(pre,perm=[0,2,3,1])
    print(pre.shape)
    target=paddle.to_tensor(target)

    
    criterion=paddle.nn.CrossEntropyLoss()

    res=criterion(pre,target)
    return res.numpy()

if __name__=="__main__":    
    from reprod_log import ReprodLogger
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    shapes=[1,20,768,768]
    pre=np.random.randn(1,20,768,768)
    target =np.random.randint(0,20,(1,768,768))

    pytorch_res=torchRes(pre,target)
    print('---pytorch---')
    print(pytorch_res)
    print(pytorch_res)
    print('---pytorch end---')
    paddle_res=paddleRes(pre,target)
    
    reprod_log_1.add("loss", pytorch_res)
    reprod_log_1.save("loss_1.npy")

    reprod_log_2.add("loss", paddle_res)
    reprod_log_2.save("loss_2.npy")