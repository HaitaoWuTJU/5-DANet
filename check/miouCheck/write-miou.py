import os
import numpy as np
np.random.seed(1)
import torch,paddle
def get_matraix(pre,target):
    
    pre=pre.reshape(-1)
    target=target.reshape(-1)
    assert(pre.shape[0]==target.shape[0])
    mat=np.zeros((20, 20), dtype=int)
    for (p1,p2) in zip(target,pre):
        mat[p1,p2]+=1
    return mat
    
def get_score(mat):
    ious=[]
    for i in range(20):
        iou=mat[i][i]/(np.sum(mat[i],axis=0)+np.sum(mat[:,i],axis=0)-mat[i][i])
        ious.append(iou)

    miou=np.mean(iou)
    ious.append(miou)
    return np.array(ious)
def paddleRes(pre,target):

    
    pre=paddle.to_tensor(pre)
    target=paddle.to_tensor(target)

    pre=pre.numpy().argmax(axis=1).astype(np.uint8)

    target=target.numpy()

    pre_mat=get_matraix(pre,target)
    paddle_res=get_score(pre_mat)

    return paddle_res
def torchRes(pre,target):
    pre=torch.from_numpy(pre).cuda()
    target=torch.from_numpy(target).cuda().long()

    pre=pre.data.cpu().numpy().argmax(axis=1).astype(np.uint8)
    print(pre.shape)
    target=target.data.cpu().numpy()
   
    pre_mat=get_matraix(pre,target)
    pytorch_res=get_score(pre_mat)
    
    return pytorch_res
if __name__=="__main__":
        
    from reprod_log import ReprodLogger
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()

    pre=np.random.randn(2,20,768,768)
    target =np.random.randint(0,20,(2,1,768,768))

    paddle_res=paddleRes(pre,target)
    pytorch_res=torchRes(pre,target)

    reprod_log_1.add("miou", pytorch_res)
    reprod_log_1.save("miou_1.npy")

    reprod_log_2.add("miou", paddle_res)
    reprod_log_2.save("miou_2.npy")