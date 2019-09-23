import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from utilities import *
from PIL import Image
from models.yolo_v1 import Yolo_V1

if __name__ == "__main__":
    imagenet_config = "./models/yolov1.cfg"
    model = Yolo_V1(imagenet_config)
    model.load_weights("./models/yolov1.weights")
    model.eval()
    x = torch.randn(2,3,448,448)
    # x = x.unsqueeze(0)
    res = model(x)    
    print(res.size())
    res = torch.flatten(res).view(x.size()[0],-1)
    print(res.size())
    print(res.numel())
