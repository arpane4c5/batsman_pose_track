#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:42:28 2019

@author: arpan

@Description: Batsman pose tracking
"""

import cv2
import numpy as np
import os
import sys
import utils

# Set the path of the input videos and the openpose extracted features directory
# Server Paths
DATASET = "/opt/datasets/cricket/ICC_WT20"
POSE_FEATS = "/home/arpan/cricket/output_json"
LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"

# Local Paths
if not os.path.exists(DATASET):
    DATASET = "/home/hadoop/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    POSE_FEATS = "/home/hadoop/VisionWorkspace/Cricket/batsman_pose_track/output_json"
    LABELS = "/home/hadoop/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"


if __name__ == '__main__':
    
    all_poses = os.listdir(POSE_FEATS)[:5]    # taking only 5 json files
    
    # get video names and divide the files into train, val, test 
#    all_files = sorted(os.listdir(DATASET))
#    train_vids, val_vids, test_vids = all_files[:16], all_files[16:21], all_files[21:]

    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
    
    #####################################################################
    
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    sizes = [utils.getNFrames(os.path.join(DATASET, f+".avi")) for f in train_lst]
    print("Size : {}".format(sizes))
    #hlDataset = VideoDataset(tr_labs, sizes, seq_size=SEQ_SIZE, is_train_set = True)
    #print hlDataset.__len__()
    
    