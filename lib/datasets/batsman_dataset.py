#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:56:54 2020

@author: arpan

@Description: BatsmanDetectionDataset for tracking. Takes input from the image files.
"""

import os
import torch
import torch.utils.data
import numpy as np
from PIL import Image
from collections import Counter

class BatsmanDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, gt_path, transforms=None):
        self.root = root
        self.transforms = transforms
        self.gt_path = gt_path
        # read all files and find unique video names
        all_files = os.listdir(root)
        #all_files_set = list(set([f.rsplit("_", 1)[0] for f in all_files]))  #unique video prefixes
        # get number of frames in each video in dictionary
        all_files_dict = dict(Counter([f.rsplit("_", 1)[0] for f in  all_files]))   
        #print(all_files_dict)
        self.img_paths = [key+"_{:012}".format(i)+".png" for key in \
                          sorted(list(all_files_dict.keys())) \
                          for i in range(all_files_dict[key])]
        self.bboxes = self.get_annotation_boxes(all_files_dict)
        
        self.bboxes_pos = []
        self.img_paths_pos = []
        for idx, box in enumerate(self.bboxes):
            if box!=[]:
                self.bboxes_pos.append(box)
                self.img_paths_pos.append(self.img_paths[idx])
                
        self.bboxes = self.bboxes[:100]
        self.img_paths_pos = self.img_paths_pos[:100]
        
        
    def get_annotation_boxes(self, keys_dict):
        ''' Create list of boxes for all the frames in the dataset.
        '''
        boxes = []
        # Iterate the video frames in the same order as done for img_paths
        for key in sorted(list(keys_dict.keys())):
            vid_nFrames = keys_dict[key]
            
            with open(os.path.join(self.gt_path, key+"_gt.txt"), "r") as fp:
                f = fp.readlines()
            
            # # remove \n at end and split into list of tuples
            # eg. tuple is ['98', '1', '303', '28', '353', '130', 'Batsman']
            f = [line.strip().split(',') for line in f]   
            f.reverse()
            frame_label = None
            
            for i in range(vid_nFrames):
                if frame_label == None:
                    if len(f) > 0:
                        frame_label = f.pop()
                    
                if frame_label is not None and int(frame_label[0])==i and \
                    int(frame_label[1])==1 and frame_label[-1]=='Batsman':
                    xmin = int(frame_label[2])
                    ymin = int(frame_label[3])
                    xmax = int(frame_label[4])
                    ymax = int(frame_label[5])
                    boxes.append([xmin, ymin, xmax, ymax])
                    frame_label = None
                else:
                    boxes.append([])
                    
        return boxes
        
        
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.img_paths_pos[idx])
        img = Image.open(img_path).convert("RGB")

        fr = img_path.rsplit(".", 1)[0].rsplit("_", 1)[1]
        fr_id = int(fr)
        
        box = self.bboxes_pos[idx]
        boxes = []
        num_objs = 1   # for only Batsman
        #if box!=[]:
        #    boxes.append(box)
        #    num_objs = 0
        boxes.append(box)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        frame_id = torch.tensor([fr_id])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["frame_id"] = frame_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths_pos)


class BatsmanDatasetFirstFrames(torch.utils.data.Dataset):
    def __init__(self, train_lst, batsman_poses_gt, poses_df, ds_path, \
                 transforms=None):
        self.train_lst = train_lst
        self.batsman_poses_gt = batsman_poses_gt
        
        bboxes_gt = []
        indexes_lst = list(poses_df.index)
        
        vids = [idx.rsplit('_', 2) for idx in indexes_lst]
        vidnames_lst = [v[0] for v in vids]
        frame_nos = [int(v[1]) for v in vids]
        
        for i, frm_no in enumerate(frame_nos):
            # find index of vidname in train_lst
            vid_idx = train_lst.index(vidnames_lst[i])
            assert vidnames_lst[i] in batsman_poses_gt[vid_idx].keys(), \
                "Frame {} not in GT {}".format(frm_no, train_lst[vid_idx])
            bboxes_gt.append(self.batsman_poses_gt[vid_idx][vidnames_lst[i]][frm_no])
            
        self.bboxes_gt = bboxes_gt
        self.indexes_lst = indexes_lst
        self.frame_nos = frame_nos
        self.vidnames_lst = vidnames_lst
        
        # TODO: How to deal with false positive and false negatives 
        
    def get_evaluation_frame(self, idx, rowname):
        return np.zeros((3, 360, 640))
        
        
    def __getitem__(self, idx):
        # load images and masks
        rowname = self.indexes_lst[idx]
        box = self.bboxes_gt[idx]
        fr_id = self.frame_nos[idx]
        num_objs = 1   # for only Batsman
        img = torch.as_tensor(self.get_evaluation_frame(idx, rowname))
        
        boxes = torch.as_tensor([box], dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        frame_id = torch.tensor([fr_id])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["frame_id"] = frame_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        return img, target, rowname
        

    def __len__(self):
        return len(self.frame_nos)
    