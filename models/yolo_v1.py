"""
TODO:
- [ ] Save the model every nth epoch
- [ ] For inference, print the time taken, classifier confidence
- [ ] Yolo class Forward function: For each pass through the network, show the layer name, num filters, filter size, input and output sizes
    - output size is calc as (h-f+1) * (w-f+1) * n_c'
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from .utilities import *
from PIL import Image
from pprint import pprint
from collections import OrderedDict

class Yolo_V1(nn.Module):
    def __init__(self, config):
        super(Yolo_V1,self).__init__()
        self.blocks = self.load_config(config)        
        self.conv_layers, self.linear_layers = self.parse_config(self.blocks)
        self.num_bound_boxes = 2

    def forward(self, x):        
        actv = x
        for i in range(len(self.conv_layers)):
            actv = self.conv_layers[i](actv)
        lin_inp = torch.flatten(actv)
        lin_inp = lin_inp.view(x.size()[0],-1) #resize it so that it is flattened by batch             
        lin_out = self.linear_layers(lin_inp)        
        det_tensor = lin_out.view(-1,((self.num_bound_boxes*5) + self.num_classes),self.grid,self.grid)
        return det_tensor #torch.flatten(det_tensor)

    def conv(self, item, prev_filters, filters, batch_norm): 
        padding = int(item['pad'])
        kernel_size = int(item['size'])
        stride = int(item['stride'])               
        bias = False if batch_norm else True
        pad = (kernel_size - 1) // 2 if padding else 0
        conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
        return conv
        
    def parse_config(self, blocks):
        conv_layers = nn.ModuleList()              
        prev_filters = 3 #image of 3 channels        
        for idx, item in enumerate(blocks[1:-2]):
            module = nn.Sequential()
            if item['type'] == 'convolutional': 
                #Add conv block           
                name = f"{item['type']}_{str(idx)}"
                filters = int(item['filters'])
                activation = item['activation'] 
                try:
                    batch_norm = bool(item['batch_normalize'])
                except:
                    batch_norm = False
                conv = self.conv(item, prev_filters, filters, batch_norm)
                module.add_module(name, conv)
                prev_filters = filters
                # Add batch_norm block
                if batch_norm:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module(f"batch_norm_{idx}",bn)
                if activation == 'leaky':
                    act = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module(f"leaky_{idx}", act)                 
            elif item['type']  == 'maxpool':
                maxpool = nn.MaxPool2d(int(item['size']), int(item['stride']))
                module.add_module(f"maxpool_{idx}", maxpool)
                # print(module[:])
            elif item['type'] == 'local':
                filters = int(item['filters'])
                activation = item['activation']
                conv = self.conv(item, prev_filters, filters, False )
                module.add_module(f"last_conv_{idx}", conv)
                if activation == 'leaky':
                    act = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module(f"leaky_{idx}", act)                
                prev_filters = filters
            elif item['type'] == 'dropout':
                module = nn.Dropout2d(p=float(item['probability']))                          
            conv_layers.append(module)                

        #Add the linear layers
        lin_item = self.blocks[-2]        
        det_item = self.blocks[-1]
        assert(lin_item['type'] == 'connected')
        l1_output = int(lin_item['output'])
        self.grid = int(det_item['side'])
        self.num_classes = int(det_item['classes'])
        num_bound_boxes = 2
        linear_layers = nn.Sequential(
            nn.Linear(7*7*256,l1_output,True),
            nn.Linear(l1_output, self.grid*self.grid * ((num_bound_boxes*5) + self.num_classes) )
        )
        return (conv_layers, linear_layers)
    

    def load_config(self, config):
        return parse_config(config)        

