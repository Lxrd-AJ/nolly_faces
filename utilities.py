from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import cv2
from torch.autograd import Variable
from math import floor, ceil

def load_classes(names):
    fp = open(names,"r")
    names = fp.read().split("\n")[:-1]
    return names

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0],3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    return canvas

def bbox_iou(box1, box2):
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    #coords of the intersecting rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    #intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    #union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)    
    prediction = prediction * conf_mask    
    box_corner = prediction.new(prediction.shape)
    #change the (center x, center y, height, width) attributes of our boxes, to 
    #(top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y).
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)
    write = False
    for idx in range(batch_size):
        image_pred = prediction[idx]
        #confidence thresholding
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes],1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)        
        seq = (image_pred[:,:5],max_conf, max_conf_score)
        image_pred = torch.cat(seq,1)
        #remove the zero rows
        non_zero_idx = torch.nonzero(image_pred[:,4])
        try:
            image_pred_ = image_pred[non_zero_idx.squeeze(),:].view(-1,7)
        except:
            continue
        img_classes = unique(image_pred_[:,-1])        
        #non-maximum suppression (nms)
        for cat in img_classes:
            cat_mask = image_pred_ * (image_pred_[:,-1] == cat).float().unsqueeze(1)
            class_mask_idx = torch.nonzero(cat_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_idx].view(-1,7)
            #sort the detections such that the entry with the maximum objectness 
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:])
                except (ValueError, IndexError) as e:
                    break
                iou_mask = (ious < nms_conf).float().unsqueeze(1)                
                image_pred_class[i+1:] *= iou_mask
                #remove the non-zero entities
                non_zero_idx = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_idx].view(-1,7)
            batch_idx = image_pred_class.new(image_pred_class.size(0),1).fill_(idx)
            seq = batch_idx, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output,out))
    try:
        return output
    except:
        return 0


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = int(inp_dim / (inp_dim / prediction.size(2))) #done like this to ensure correct results
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    assert(grid_size == prediction.size(2))

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #sigmoid the centre_x, centre_y and objectness confidence scores   
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    x_offset = x_offset.cuda() if CUDA else x_offset
    y_offset = y_offset.cuda() if CUDA else y_offset

    x_y_offset = torch.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset
    anchors = torch.FloatTensor(anchors).cuda() if CUDA else torch.FloatTensor(anchors)
    anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]) * anchors

    #sigmoid activation to the total class scores
    prediction[:,:,5:5+num_classes] = torch.sigmoid(prediction[:,:,5:5+num_classes])
    #resize the detections map to the size of the input image
    prediction[:,:,:4] *= stride

    return prediction
