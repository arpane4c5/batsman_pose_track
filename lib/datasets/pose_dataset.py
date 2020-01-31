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
    def __init__(self, ds_path, pose_path, train_lst, tr_labs, epsilon=50, bboxes=False, \
                 transforms=None):
        '''
        
        '''
        self.ds_path = ds_path
        self.pose_path = pose_path
        self.train_lst = train_lst
        self.tr_labs = tr_labs
        
        track_lst, count = track_stroke_poses(ds_path, pose_path, train_lst, tr_labs, \
                                              epsilon, bboxes, False)
        
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
        #self.indexes_lst = indexes_lst
        self.frame_nos = frame_nos
        self.vidnames_lst = vidnames_lst
        
        # TODO: How to deal with false positive and false negatives 
        
    def get_evaluation_frame(self, idx, rowname):
        return np.zeros((3, 360, 640))
    
    
def track_stroke_poses(datasetPath, pose_feats, train_lst, tr_labs, epsilon=50, \
                       bboxes=True, visualize=True):
    '''
    Tracking the poses using the VideoPose class and TrackPose
    
    '''
    track_lst, track_count = [], 0
    for i, vid_file in enumerate(train_lst):
        # Create object for one video only
        v = VideoPoseTrack(datasetPath, train_lst[i], pose_feats, tr_labs[i])
        # Visualize the bounding boxes or keypoints of frame poses, without tracking
#        v.visualizeVideoWithPoses(bboxes=bboxes)
        # detect tracks based on nearest neighbours with euclidean dist < epsilon
        vid_tracks, count = v.track_poses(epsilon=epsilon, bboxes=bboxes, \
                                          visualize=visualize)
        track_lst.append(vid_tracks)
        track_count += count
        
        lengths, pids, strokes = v.find_longest_track(epsilon, bboxes, visualize)
        for i, pid in enumerate(pids):
            print("Longest track : Pose ID : {} :: Length : {} :: Stroke : {}"\
                  .format(pid, lengths[i], strokes[i]))
            
        normed_poses = v.get_normalized_poses()

    return track_lst, track_count

def pad_tensor(vec, pad, value=0, dim=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = pad - vec.shape[0]

    if len(vec.shape) == 2:
        zeros = torch.ones((pad_size, vec.shape[-1])) * value
    elif len(vec.shape) == 1:
        zeros = torch.ones((pad_size,)) * value
    else:
        raise NotImplementedError
    return torch.cat([torch.Tensor(vec), zeros], dim=dim)
    
def pad_collate(batch, values=(0, 0), dim=0):
    """
    args:
        batch - list of (tensor, label)

    reutrn:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
        ws - a tensor of sequence lengths
    """

    sequence_lengths = torch.Tensor([int(x[0].shape[dim]) for x in batch])
    sequence_lengths, xids = sequence_lengths.sort(descending=True)
    target_lengths = torch.Tensor([int(x[1].shape[dim]) for x in batch])
    target_lengths, yids = target_lengths.sort(descending=True)
    # find longest sequence
    src_max_len = max(map(lambda x: x[0].shape[dim], batch))
    tgt_max_len = max(map(lambda x: x[1].shape[dim], batch))
    # pad according to max_len
    batch = [(pad_tensor(x, pad=src_max_len, dim=dim), pad_tensor(y, pad=tgt_max_len, dim=dim)) for (x, y) in batch]

    # stack all
    xs = torch.stack([x[0] for x in batch], dim=0)
    ys = torch.stack([x[1] for x in batch]).int()
    xs = xs[xids]
    ys = ys[yids]
    return xs, ys, sequence_lengths.int(), target_lengths.int()