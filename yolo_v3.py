from __future__ import division
from torch.autograd import Variable
from pprint import pprint
from PIL import Image
from utilities import *
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
    upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    module.add_module(f"upsample_{idx}", upsample)


def add_module_route(module,x,idx,output_filters):
    x['layers'] = x['layers'].split(",")
    start = int(x['layers'][0])
    end = int(x['layers'][1]) if len(x['layers']) > 1 else 0    
    #positive annotations
    start = start - idx if start > 0 else start
    end = end - idx if end > 0 else end    
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
        outputs = {}
        write = 0
        for idx, module in enumerate(modules):
            module_type = module["type"]            
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[idx](x)                
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]                
                if layers[0] > 0:
                    layers[0] = layers[0] - idx
                if len(layers) == 1:
                    x = outputs[idx+layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - idx
                    map1 = outputs[idx + layers[0]]
                    map2 = outputs[idx + layers[1]]
                    x = torch.cat((map1,map2),1)
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[idx-1] + outputs[idx+from_]
            elif module_type == "yolo":
                anchors = self.module_list[idx][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])
                x = x.data
                x = predict_transform(x,inp_dim,anchors,num_classes,CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections,x),1)
            outputs[idx] = x
        return detections

    def load_weights(self, wfile):
        with open(wfile,'rb') as file:
            header = np.fromfile(file, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(file, dtype=np.float32)
            ptr = 0
            for i in range(len(self.module_list)):
                module_type = self.blocks[i+1]["type"]
                if module_type == "convolutional":
                    model = self.module_list[i]
                    try:
                        batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                    except:
                        batch_normalize = 0
                    conv = model[0]
                    if batch_normalize:
                        bn = model[1]
                        num_bn_biases = bn.bias.numel()
                        bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        ptr += num_bn_biases

                        bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        ptr += num_bn_biases

                        bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        ptr += num_bn_biases

                        bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        ptr += num_bn_biases

                        #cast the loaded weights into dims of model weights
                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weights = bn_weights.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_var = bn_running_var.view_as(bn.running_var)
                        #copy the data to the model
                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)
                    else:
                        num_biases = conv.bias.numel()
                        conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                        ptr += num_biases
                        conv_biases = conv_biases.view_as(conv.bias.data)
                        conv.bias.data.copy_(conv_biases)
                    
                    num_weights = conv.weight.numel()
                    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                    ptr += num_weights
                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)



def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


if __name__ == "__main__":
    # blocks = parse_config("./cfg/yolov3.cfg")
    # net_info, modules = create_modules_from( blocks )
    # count = 0
    # for ch in modules.children():
    #     print(ch)
    #     count += 1
    # print(count)
    
    _IMAGE_SIZE_ = (800,800)
    sample_input = Image.open('data.png')
    sample_input.thumbnail(_IMAGE_SIZE_, Image.ANTIALIAS)
    # sample_input.show()
    sample_input = np.array(sample_input, dtype=np.uint8)
    sample_input = torch.from_numpy( sample_input.transpose((2,0,1)) )
    sz = sample_input.size()    
    sample_input = sample_input.contiguous().view(1,sz[0],sz[1],sz[2]).type(torch.FloatTensor)

    model = YOLO("./cfg/yolov3.cfg")
    model.load_weights("yolov3.weights")

    inp = get_test_input()
    print("Input size", inp.size())
    pred = model(inp,torch.cuda.is_available())
    print(pred)
    print("Prediction size", pred.size())
    _ = write_results(pred, 0.7, 80)
    
