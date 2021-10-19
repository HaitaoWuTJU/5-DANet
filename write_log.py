import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
import DANetPaddle.work.paddle.danet.networks.danet as danet
import DANetPytorch.encoding as encoding
import torch,os
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
    reprod_log_1.save("res_1.npy", pytorch_res1)
    reprod_log_1.save("res_2.npy", paddle_res1)


    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("./res_1.npy")
    info2 = diff_helper.load_info("./res_2.npy")

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./diff.txt")
