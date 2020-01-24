#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:56:25 2020

@author: arpan

@Description: PoseDataset class for extracting the pose based features
"""

import _init_paths
import os
import torch
import numpy as np
from lib.models.video_pose import VideoPoseTrack


class PoseDataset(torch.utils.data.Dataset):
    '''
    Generate pose sequence segments.
    '''
    def __init__(self, ds_path, pose_path, train_lst, tr_labs, strokes, bboxes=False, \
                 transforms=None):
        '''
        
        '''
        self.ds_path = ds_path
        self.pose_path = pose_path
        self.train_lst = train_lst
        self.tr_labs = tr_labs
        
        poses_lst = []
        batsman_poses_gt = []
        for i, vid_file in enumerate(train_lst):
            # Create object for one video only
            v = VideoPoseTrack(ds_path, train_lst[i], pose_path, tr_labs[i])
            
            # get list of pose matrices for target frames by iterating on strokes
            # each sublist is for a stroke with n arrays for n first frames
            topn_poses = v.getCAMFramePoses(first_n = n, bounding_box=bboxes)
            #batsman_poses_gt.append({vid_file : utils.getGTBatsmanPoses(BAT_LABELS, \
            #                                                    tr_gt_batsman[i])})
            #v.visualizeVideoWithBatsman()
            # Iterate on the strokes to get the frameNo and PoseNo for rownames
            for stroke_idx, (beg, end) in enumerate(v.strokes):
                for frm_no in range(n):     # Iterate over first n frames of each stroke
                    # get the matrix of poses for a selected frame
                    pose_arr = topn_poses[stroke_idx][frm_no]
                    rownames = []
                    for pose_no in range(pose_arr.shape[0]):    # Iterate on the poses
                        rownames.append(train_lst[i]+"_"+str(beg+frm_no)+"_P"+str(pose_no))
                    # create a sub-dataframe and append to list
                    #poses_lst.append(pd.DataFrame(pose_arr, index=rownames))
        #return pd.concat(poses_lst), batsman_poses_gt     # vstack all sub-dataframes
        
        # A sequence of vectors will be using BoPW 
        
        
        bboxes_gt = []
        #indexes_lst = list(.index)
        
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