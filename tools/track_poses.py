#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:42:28 2019

@author: arpan

@Description: Batsman pose tracking
"""

import _init_paths

import cv2

import os
import numpy as np
import pandas as pd
from utils import track_utils as utils
from lib.models.video_pose import VideoPoseTrack
from visualize import visualize_poses as viz

from scipy.spatial import distance
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.vq import vq

from evaluation.engine import evaluate
from datasets.batsman_dataset import BatsmanDatasetFirstFrames
import torch


# Set the path of the input videos and the openpose extracted features directory
# Server Paths
DATASET = "/opt/datasets/cricket/ICC_WT20"
POSE_FEATS = "/home/arpan/cricket/output_json"
BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
#TRAIN_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/train"
#VAL_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/val"
#TEST_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/test"
#ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"

# Local Paths
if not os.path.exists(DATASET):
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    POSE_FEATS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/data/output_json"
    BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"


def get_stacked_poses_df(train_lst, tr_labs, tr_gt_batsman, bboxes=True, n=1):
    '''
    Create a dataframe of the first n frame poses with "vidname_frameNo_P0" as rownames.
    Total number of rows are \Sigma_{i} \Sigma_{j} {P_{ij}} where j is pose in each frame
    and i iterates on selected frames. For n=1 i iterates on only first frames of each
    stroke.
    
    Parameters:
    ----------
    train_lst : list of str
        list of video file names for which poses have to be extracted
    tr_labs : list of str
        list of paths of corresponding json files having stroke labels
    tr_gt_batsman : list of str
        list of paths of corresponding txt files having batsman locations
    bboxes : boolean
        True if bounding boxes have to be extracted, False if 75 dimensional 
        pose vectors have to be extracted
    n : int
        No. of frames as starting of the stroke that have to be considered for 
        extraction
        
    Returns:
    -------
    pd.DataFrame : an (P x 75) or (P x 4) dataframe with rownames as 
    vidname_frmNo_PoseNo where P is total number of poses in set of frames
    
    '''
    poses_lst = []
    batsman_poses_gt = []
    for i, vid_file in enumerate(train_lst):
        # Create object for one video only
        v = VideoPoseTrack(DATASET, train_lst[i], POSE_FEATS, tr_labs[i])
        
        # get list of pose matrices for target frames by iterating on strokes
        # each sublist is for a stroke with n arrays for n first frames
        topn_poses = v.getCAMFramePoses(first_n = n, bounding_box=bboxes)
        batsman_poses_gt.append({vid_file : utils.getGTBatsmanPoses(BAT_LABELS, \
                                                            tr_gt_batsman[i])})
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
                poses_lst.append(pd.DataFrame(pose_arr, index=rownames))
    return pd.concat(poses_lst), batsman_poses_gt     # vstack all sub-dataframes

        
def kmeans(poses, clusters=4, random_state=0):
    '''
    Clustering a numpy matrix or pandas dataframe into N clusters using KMeans.
    Random state is passed as argument
    Parameters:
    -----------
    flows : np.array or pd.DataFrame
        matrix with data to be clustered
    clusters : int
        No. of clusters 
        
    Returns:
    --------
    KMeans object having "clusters" cluster centers and related information.
    '''
    km = KMeans(n_clusters=clusters, algorithm='elkan', random_state=random_state)
    km.fit(poses)
    return km

def dbscan(poses, epsilon=100, min_samples=10):
    '''
    Use Density Based Spatial Clustering for applications with Noise.
    '''
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(poses)
    return db
    
    
def track_stroke_poses(train_lst, tr_labs, tr_gt_batsman, epsilon=50, bboxes=True, visualize=True):
    '''
    Tracking the poses using the VideoPose class and TrackPose
    
    '''
    track_lst, track_count = [], 0
    for vid, vid_file in enumerate(train_lst):
        # Create object for one video only
        v = VideoPoseTrack(DATASET, train_lst[vid], POSE_FEATS, tr_labs[vid])
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
        
        v.plot_tracks(BAT_LABELS, tr_gt_batsman[vid], bboxes)
        break

    return track_lst, track_count
    

if __name__ == '__main__':

    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
    # get list of label filenames containing temporal stroke and batsman labels
    train_lab = [f.rsplit('.',1)[0] +".json" for f in train_lst]
    val_lab = [f.rsplit('.',1)[0] +".json" for f in val_lst]
    test_lab = [f.rsplit('.',1)[0] +".json" for f in test_lst]
    train_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in train_lst]
    val_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in val_lst]
    test_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in test_lst]
    
    #####################################################################
    
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    tr_gt_batsman = [os.path.join(BAT_LABELS, f) for f in train_gt_batsman]
    sizes = [utils.getTotalFramesVid(os.path.join(DATASET, f)) for f in train_lst]
    print("Size : {}".format(sizes))
    
    bboxes = True
    epsilon = 50
    visualize = False
    n_clusters = 5
    clust_no = 1
    first_frames = 2
    
#    ###########################################################################
#    # 1: Extraction of batsman poses and visualize on the frames and plots
#    
#    poses_df, batsman_gt_train = get_stacked_poses_df(train_lst, tr_labs, \
#                                            train_gt_batsman, bboxes, n=first_frames)
##    poses_df.to_csv("poses_df_nFrames1.csv")
#    
##    poses_df = pd.read_csv("poses_df_nFrames1.csv", index_col=0)
#    
##    nboxes, avg_w, avg_h, area = utils.get_avg_box_values(batsman_gt_train)
#        
#    
#    km_poses = kmeans(poses_df, clusters=n_clusters)
#    
#    clust_centers = km_poses.cluster_centers_
#    word_ids = vq(poses_df, clust_centers)[0]  # ignoring the distances in [1]
#    
#    # Cluster the PoseVectors and consider 0 keypoints as mean of available ones
#    
#    # Compute accuracy by evaluating mAP for sequence of frames
#
##    viz.visualize_prediction(DATASET, poses_df.iloc[word_ids==clust_no], clust_no, bboxes)
##    
##    viz.plot_predictions_mask(DATASET, poses_df, word_ids, n_clusters, bboxes)
##
##    
##    pose_path_lst = viz.writeCAMFramePoses(DATASET, poses_df, "extracted_poses", write=False)
##    
##    viz.plot_predictions_interactive(poses_df, word_ids, n_clusters, pose_path_lst, bboxes)
#
#    # clust_no = 2 is for persons near and corresponding to batsman
#    
#    poses_df_val, batsman_poses_gt = get_stacked_poses_df(val_lst, val_labs, \
#                                            val_gt_batsman, bboxes, n=first_frames)
#    
#    word_ids_val = vq(poses_df_val, clust_centers)[0]
##    # infer on the validation set features
#    
#    
#    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#    device = torch.device('cpu')
#    
#    for cl in [1]:
#        print("Cluster No {}".format(cl))
#        # Create a dataset loader or dataset
#        dataset_test = BatsmanDatasetFirstFrames(val_lst, batsman_poses_gt, \
#                                poses_df_val.iloc[word_ids_val==cl], DATASET)
#        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, \
#                            shuffle=False, num_workers=1, collate_fn=utils.collate_fn)
#        #viz.visualize_prediction(DATASET, poses_df.iloc[word_ids==cl], cl, bboxes)
#        evaluate(poses_df_val.iloc[word_ids_val==cl], data_loader_test, device)
#    
#    ###########################################################################
    
    # 2: Track the poses for strokes
    track_lst, count = track_stroke_poses(train_lst, tr_labs, tr_gt_batsman, epsilon, bboxes, visualize)
    
    #plot_tracks(bboxes)
    
    