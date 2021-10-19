import numpy as np
from reprod_log import ReprodLogger
import DANetPaddle.work.paddle.danet.networks.danet as danet
import DANetPytorch.encoding as encoding
import torch,os
import sys
sys.path.append('/home/haitaowu/5-DANet/DANetPaddle')
sys.path.append('/home/haitaowu/5-DANet/DANetPytorch')

import paddle
if __name__ == "__main__":
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()

    data_1 = np.random.rand(1, 3, 768, 768).astype(np.float32)
    data_2 = np.random.rand(1, 3, 768, 768).astype(np.float32)

    modelPytorch = encoding.models.get_model('resnest101', pretrained=False)

    state = {'net':modelPytorch.state_dict()}
    torch.save(state,os.path.join(os.getcwd(),'torch.pth'))
    
    modelPaddle=danet.DANet('danet')
    
    tmp=torch.from_numpy(data_1)
    pytorch_res1=modelPytorch(tmp)
    tmp=paddle.to_tensor(data_1)
    paddle_res1=modelPaddle(tmp)
    

    reprod_log_1.add("pytorch_model_1", pytorch_res1)
    reprod_log_1.add("paddle_model2_1", paddle_res1)
    reprod_log_1.save("model_1.npy")

    reprod_log_2.add("pytorch_model_2", data_2)
    reprod_log_2.add("paddle_model_2", data_2)
    reprod_log_2.save("model_2.npy")