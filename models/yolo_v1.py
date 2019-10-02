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
        self.num_bbox = 2
        self.input_size = (448,448)
        self.num_classes = -1
        self.blocks = self.load_config(config)        
        self.conv_layers, self.linear_layers = self.parse_config(self.blocks)
        

    def forward(self, x):        
        actv = x
        for i in range(len(self.conv_layers)):
            actv = self.conv_layers[i](actv)
            block_info = self.blocks[i+1]            
            assert not torch.isnan(actv).any()

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
            pred_classes = convert_cls_idx_name(self.cls_names, max_idx.numpy())
                                    
            bboxes = torch.cat((bboxes, max_idx.unsqueeze(1).float()),1)            
            bboxes = confidence_threshold(bboxes, 0.5) # confidence thresholding            
            #TODO: Continue; Non-maximum suppression for each class
            results[batch] = bboxes
        
        return results
            


    def build_class_map(self, fname):        
        fp = open(fname,"r")
        names = fp.read().split("\n")
        names_dict = {idx:e for idx, e in enumerate(names)}
        self.cls_names = names_dict
        return names_dict
            

    def load_weights(self, weights_file):
        """TODO: Fix bug here
        Either i am using the wrong weights file or I am not reading from this weight file
        properly. Either way, there is a lot of unread weights values left in the file.
        
        TODO: Break down this function into smaller blocks
        """
        with open(weights_file,'rb') as file:
            #NB: An internal file pointer is maintained, so read the header first
            header = np.fromfile(file, dtype=np.int32, count=5)
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
                        

    def conv(self, item, prev_filters, filters, batch_norm): 
        padding = int(item['pad'])
        kernel_size = int(item['size'])
        stride = int(item['stride'])               
        bias = False if batch_norm else True
        pad = (kernel_size - 1) // 2 if padding else 0
        conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
        return conv
        
    def parse_config(self, blocks):
        """
        TODO: Break down this function into smaller blocks
        """
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

