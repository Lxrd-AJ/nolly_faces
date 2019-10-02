import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import cv2 as cv
from torch.autograd import Variable
from models.utilities import *
from PIL import Image, ImageOps
from models.yolo_v1 import Yolo_V1

if __name__ == "__main__":
    imagenet_config = "./models/yolov1.cfg"
    model = Yolo_V1(imagenet_config)
    model.load_weights("./models/yolov1.weights")
    _ = model.build_class_map("./models/voc.names")    
    # model.eval() #TODO: Model Batch size affects the activations, it introduces NaNs into the results
    
    # Load the test image
    # x = torch.randn(3,3,448,448) # x = x.unsqueeze(0)
    
    x = Image.open("./grab_drive_3_1.png")    
    x = x.resize((448,448)) #TODO: Research how to maintain the aspect ratio
    # x.show()
    x = np.array(x)
    x_np = np.copy(x)
    x = x[:,:,:3].transpose((2,0,1))
    x = torch.from_numpy(x).float().div(255).unsqueeze(0)   

    x = Variable(x) 
    assert not torch.isnan(x).any()
    
    # print("Input size", x.size())
    
    with torch.no_grad():
        res = model(x)        
    # print("Prediction size", res.size())
    res = model.transform_predict(res)
    class_names = convert_cls_idx_name(model.cls_names, res[0][:,5].numpy())
    res = res[0].numpy()
    #convert the width and height to bottom-right coords
    res[:,3] = res[:,1] + res[:,3]
    res[:,4] = res[:,2] + res[:,4]
    for i in range(res.shape[0]):
        box = res[i,:]        
        print(box[1:-1])
    # res = torch.flatten(res).view(x.size()[0],-1)
    # print("Flattened prediction",res.size())
    # print("Num elements in prediction",res.numel())
