"""
TODO:
- [ ] Save the model every nth epoch
- [ ] For inference, print the time taken, classifier confidence
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
    def __init__(self, class_names, grid_size, blocks):
        super(Yolo_V1,self).__init__()
        self.num_bbox = 2
        self.input_size = (448,448)
        self.class_names = class_names
        self.num_classes = len(class_names.keys())        
        self.grid = grid_size        
        self.extraction_layers, extract_out = self.parse_conv(blocks)
        self.final_conv = nn.Conv2d(extract_out, 256, 3, 1, 1)
        self.linear_layers = nn.Sequential(
            nn.Linear(12*12*256,1715,True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1715, self.grid*self.grid * ((self.num_bbox*5) + self.num_classes)),
            nn.ReLU(inplace=True)
        )        
        
    def forward(self, x):        
        actv = x
        for i in range(len(self.extraction_layers)):
            actv = self.extraction_layers[i](actv)                       
            assert not torch.isnan(actv).any()

        actv = self.final_conv(actv)

        lin_inp = torch.flatten(actv)
        lin_inp = lin_inp.view(x.size()[0],-1) #resize it so that it is flattened by batch             
        lin_out = self.linear_layers(lin_inp)
        lin_out = torch.sigmoid(lin_out)            
        det_tensor = lin_out.view(-1,((self.num_bbox * 5) + self.num_classes),self.grid,self.grid)
        return det_tensor #torch.flatten(det_tensor)

    def transform_predict(self, p_tensor):
        batch_size = p_tensor.size(0)
        stride = self.input_size[0] // p_tensor.size(2)
        grid_size = self.input_size[0] // stride
        num_bbox = (self.num_bbox * 5) + self.num_classes
        predictions = p_tensor.view(batch_size, num_bbox, grid_size*grid_size)
        predictions = predictions.transpose(1,2).contiguous()
        num_bbox = 5 + self.num_classes
        
        results = {}
        for batch in range(predictions.size(0)):
            prediction = predictions[batch]
                        
            bboxes = prediction[:,:10]
            bbox_1 = convert_center_coords_to_noorm( bboxes[:,:5] )
            bbox_2 = convert_center_coords_to_noorm( bboxes[:,5:] )
            bboxes = max_box(bbox_1, bbox_2)
            
            cls_probs = prediction[:,10:]
            max_cprob, max_idx = cls_probs.max(1) #1 is along the rows            
            pred_classes = convert_cls_idx_name(self.class_names, max_idx.numpy())

            bboxes = torch.cat((bboxes, max_idx.unsqueeze(1).float()),1)            
            bboxes = confidence_threshold(bboxes, 0.5) # confidence thresholding            
            #TODO: Continue; Non-maximum suppression for each class
            results[batch] = bboxes
        
        return results
            

    # def conv(self, item, prev_filters, filters, batch_norm): 
    #     padding = int(item['pad'])
    #     kernel_size = int(item['size'])
    #     stride = int(item['stride'])               
    #     bias = False if batch_norm else True
    #     pad = (kernel_size - 1) // 2 if padding else 0
    #     conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
    #     return conv

    """
    Returns the convolutional blocks in the YOLO module architecture
    This is based on the extraction net architecture as described here https://pjreddie.com/darknet/imagenet/#extraction
    """
    def seq_conv(self, item, module, idx, in_channels):
        padding = int(item['pad'])
        kernel_size = int(item['size'])
        stride = int(item['stride'])               
        pad = int(item['pad'])
        filters = int(item['filters'])
        activation = item['activation']
        try:
            batch_norm = bool(item['batch_normalize'])
        except:
            batch_norm = False
        bias = False if batch_norm else True

        module.add_module(f'conv_{idx}', nn.Conv2d(in_channels, filters, kernel_size, stride, pad, bias=bias))
        if batch_norm:
            module.add_module(f"batch_norm_{idx}", nn.BatchNorm2d(filters))

        if activation == 'leaky':
            module.add_module(f"leaky_{idx}", nn.LeakyReLU(0.1, inplace=True))
        else:
            print("Unknown activation function provided for YOLO v1")

        return module, filters

    def parse_conv(self, blocks):
        conv_layers = nn.ModuleList()              
        prev_filters = 3 #image of 3 channels        
        for idx, item in enumerate(blocks[1:-1]):
            module = nn.Sequential()
            
            if item['type'] == 'convolutional': 
                module, prev_filters = self.seq_conv(item, module, idx, prev_filters)
            elif item['type']  == 'maxpool':
                maxpool = nn.MaxPool2d(int(item['size']), int(item['stride']))
                module.add_module(f"maxpool_{idx}", maxpool)
            elif item['type'] == 'avgpool':
                module.add_module(f"avgpool_{idx}", nn.AvgPool2d(2,2))

            conv_layers.append(module)
        return conv_layers, prev_filters

        # print(self.blocks)
        # #Add the linear layers
        # lin_item = self.blocks[-2]        
        # det_item = self.blocks[-1]
        # assert(lin_item['type'] == 'connected')
        # l1_output = int(lin_item['output'])
        # self.grid = int(det_item['side'])
        # self.num_classes = int(det_item['classes'])
        # num_bound_boxes = 2
        # linear_layers = nn.Sequential(
        #     nn.Linear(7*7*256,l1_output,True),
        #     nn.Linear(l1_output, self.grid*self.grid * ((num_bound_boxes*5) + self.num_classes) )
        # )
        # return (conv_layers, linear_layers)      

    def load_weights(self, weights_file):
        """TODO: Fix bug here
        Either i am using the wrong weights file or I am not reading from this weight file
        properly. Either way, there is a lot of unread weights values left in the file.
        
        TODO: Break down this function into smaller blocks
        """
        with open(weights_file,'rb') as file:
            #NB: An internal file pointer is maintained, so read the header first
            header = np.fromfile(file, dtype=np.int32, count=5)
            print(header)
            weights = np.fromfile(file, dtype=np.float32)    
            idx = 0  
            # populate the convolutional layers           
            for i in range(len(self.conv_layers)):
                block_info = self.blocks[i+1]
                if block_info['type'] in ['convolutional','local']:                    
                    try:
                        batch_norm = bool(block_info['batch_normalize'])
                    except:
                        batch_norm = False
                    conv = self.conv_layers[i][0]
                    if batch_norm: #load the biases, weights, running_mean and running variance
                        bn = self.conv_layers[i][1]
                        num_bn_bias = bn.bias.numel()
                        bn_bias = torch.from_numpy(weights[idx:idx+num_bn_bias])
                        idx += num_bn_bias
                        # batch norm weights
                        num_bn_weights = bn.weight.numel()
                        assert(num_bn_bias == num_bn_weights)
                        bn_weights = weights[idx:idx+num_bn_weights]
                        idx += num_bn_bias
                        bn_weights = torch.from_numpy(bn_weights)
                        # batch norm running mean
                        assert( num_bn_bias == bn.running_mean.numel())
                        bn_running_mean = torch.from_numpy(weights[idx:idx+num_bn_bias])
                        idx += num_bn_bias
                        # running variance
                        bn_running_variance = torch.from_numpy(weights[idx:idx+num_bn_bias])
                        idx += num_bn_bias

                        #convert the loaded params into the same size as the target bias params
                        bn_bias = bn_bias.view_as(bn.bias.data)
                        bn_weights = bn_weights.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_variance = bn_running_variance.view_as(bn.running_var)
                        # copy the loaded params into the bias params
                        bn.bias.data.copy_(bn_bias)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_variance)
                    else: #copy the conv bias
                        num_conv_bias = conv.bias.numel()
                        conv_bias = torch.from_numpy(weights[idx:idx+num_conv_bias])
                        idx += num_conv_bias
                        conv_bias = conv_bias.view_as(conv.bias.data)
                        conv.bias.data.copy_(conv_bias)
                    # copy the conv weights
                    num_conv_weights = conv.weight.numel()
                    conv_weights = torch.from_numpy(weights[idx:idx+num_conv_weights])
                    idx += num_conv_weights
                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)       
                else:
                    # print(block_info)
                    pass
            # Populate the linear layers
            for k in range(len(self.linear_layers)):
                # print(self.linear_layers[k])
                linear = self.linear_layers[k]
                # copy the bias
                num_lin_bias = linear.bias.numel()
                lin_bias = torch.from_numpy(weights[idx:idx+num_lin_bias])
                lin_bias = lin_bias.view_as(linear.bias.data)
                linear.bias.data.copy_(lin_bias)
                idx += num_lin_bias
                # copy the weights
                num_lin_weights = linear.weight.numel()
                lin_weights = torch.from_numpy(weights[idx:idx+num_lin_weights])
                lin_weights = lin_weights.view_as(linear.weight.data)
                linear.weight.data.copy_(lin_weights)
                idx += num_lin_weights

            # print("Index into weights", idx)
            # print("Total weights ", len(weights))
            # print("Num weights left ", len(weights) - idx)
            # print("---" * 15)