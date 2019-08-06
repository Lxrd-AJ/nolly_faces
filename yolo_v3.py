from __future__ import division
from torch.autograd import Variable
from pprint import pprint
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors


def parse_config(cfg_file):
    with open(cfg_file) as file:
        lines = file.read().split('\n')
        lines = [x for x in lines if len(x) > 0]
        lines = [x for x in lines if x[0] != '#'] #remove comments
        lines = [x.rstrip().lstrip() for x in lines]
        
        block = {}
        blocks = []

        for line in lines:
            if line[0] == '[':
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()                
            else:
                key, value = line.split("=")                
                block[key.rstrip()] = value.lstrip()
        blocks.append(block) 
        return blocks       
        
def add_module_convolutional(module, x, idx, prev_filters):
    activation = x['activation']
    try:
        batch_normalise = int(x['batch_normalize'])
        bias = False
    except:
        batch_normalise = 0
        bias = True
    filters = int(x['filters'])
    padding = int(x['pad'])
    kernel_size = int(x['size'])
    stride = int(x['stride'])
    pad = (kernel_size - 1) // 2 if padding else 0
    conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
    module.add_module(f"conv_{idx}", conv)

    if batch_normalise:
        bn = nn.BatchNorm2d(filters)
        module.add_module(f"batch_norm_{idx}", bn)
    if activation == 'leaky':
        activ = nn.LeakyReLU(0.1, inplace=True)
        module.add_module(f"leaky_{idx}", activ)

    return filters
    

def add_module_upsample(module, x, idx):
    stride = int(x['stride'])
    upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    module.add_module(f"upsample_{idx}", upsample)


def add_module_route(module,x,idx,output_filters):
    x['layers'] = x['layers'].split(",")
    start = int(x['layers'][0])
    end = int(x['layers'][1]) if len(x['layers']) > 1 else 0    
    #positive annotations
    start = start - idx if start > 0 else start
    end = end - idx if end > 0 else end
    print(idx,start,output_filters)
    route = EmptyLayer()
    module.add_module(f"route_{idx}", route)
    if end < 0:
        filters = output_filters[idx+start] + output_filters[idx+end]
    else:
        filters = output_filters[idx + start]
    return filters

def add_module_yolo(module,x,idx):
    mask = x['mask'].split(',')
    mask = [int(x) for x in mask]
    anchors = [int(a) for a in x['anchors'].split(',')]
    anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
    anchors = [anchors[i] for i in mask]
    detection = DetectionLayer(anchors)
    module.add_module(f"Detection_{idx}", detection)



def create_modules_from(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for idx, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if (x['type'] == 'convolutional'):
            filters = add_module_convolutional(module, x, idx, prev_filters)
        elif x["type"] == "upsample":
            add_module_upsample(module,x,idx)
        elif x['type'] == "route":
            filters = add_module_route(module,x,idx, output_filters)
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{idx}", shortcut)
        elif x['type'] == 'yolo':
            add_module_yolo(module,x,idx)
    
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)    

    return (net_info, module_list)



class YOLO(nn.Module):
    def __init__(self, cfg_file):
        super(YOLO,self).__init__()
        self.blocks = parse_config(cfg_file)
        self.net_info, self.module_list = create_modules_from(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]

if __name__ == "__main__":
    # blocks = parse_config("./cfg/yolov3.cfg")
    # net_info, modules = create_modules_from( blocks )
    # count = 0
    # for ch in modules.children():
    #     print(ch)
    #     count += 1
    # print(count)

    yolo = YOLO("./cfg/yolov3.cfg")
    print(yolo.blocks)
