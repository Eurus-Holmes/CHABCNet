import torch

pthfile = r'model_final.pth'
net = torch.load(pthfile, map_location=torch.device('cpu'))
for key, value in net["model"].items():
    print(key, value.size())

