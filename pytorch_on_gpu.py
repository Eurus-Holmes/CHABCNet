import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print("torch.cuda.is_available() =", torch.cuda.is_available())

print("torch.backends.cudnn.is_available() =", torch.backends.cudnn.is_available())

print("torch.backends.cudnn.version() =", torch.backends.cudnn.version())

print("torch.rand(2, 3).cuda() =", torch.rand(2, 3).cuda())
