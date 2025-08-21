import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    torch_version = torch.__version__
    flag = torch.cuda.is_available()
    ngpu = torch.cuda.device_count()
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()

    if flag and ngpu > 0:
        print(f"PyTorch Version:\t {torch_version}")
        print(f"number of GPUs:\t {ngpu}")
        print(f"\'cuda:0:\':\t {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version:\t {cuda_version}")
        print(f"cuDNN Version:\t {cudnn_version}")
        print(torch.rand(3, 3).cuda())
    else:
        print("CUDA is not available. Running on CPU.")
        print(torch.rand(3, 3))