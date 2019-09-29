import numpy as np 
import torch

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

def convert_center_coords_to_noorm(bboxes):
    (rows,cols) = (7,7)    
    stride = 64
    assert (rows * cols) == bboxes.size(0)
    grid_strides = []
    """ There might be a better way to generate this interpolation of grid sizes
        check out linspace and meshgrid in numpy
    """
    for grid in range(rows * cols):
        row = grid // rows
        row_stride = row * 64
        col = grid - (row * cols)
        col_stride = col * 64
        grid_strides.append([row_stride,col_stride])
    # center coordinates
    grid_strides = np.array(grid_strides)
    grid_strides = torch.from_numpy(grid_strides).float()
    bboxes[:,1:3] = (bboxes[:,1:3] * stride).round()
    bboxes[:,1:3] = bboxes[:,1:3] + grid_strides
    bboxes[:,3:] = (bboxes[:,3:].pow(2) * 448).round()
    # Convert x,y to top left coords
    bboxes[:,1] -= bboxes[:,3]/2
    bboxes[:,2] -= bboxes[:,4]/2
    return bboxes


def convert_cls_idx_name(name_mapping, arr):
    return [name_mapping[x] for x in arr]