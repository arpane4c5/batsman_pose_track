#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 20:34:19 2021

@author: arpan

@Description: Batsman pose tracking
"""

import _init_paths
#
import cv2
import sys
import os
import numpy as np
import json
import pickle
#import pandas as pd
from utils import track_utils as utils
from lib.models.video_pose import VideoPoseTrack
#from visualize import visualize_poses as viz

#from scipy.spatial import distance
#from sklearn.cluster import KMeans, DBSCAN
#from scipy.cluster.vq import vq
#
#from evaluation.engine import evaluate
#from datasets.batsman_dataset import BatsmanDatasetFirstFrames


# Set the path of the input videos and the openpose extracted features directory
# Server Paths
#TRAIN_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/train"
#VAL_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/val"
#TEST_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/test"
#ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"

# Local Paths
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
POSE_FEATS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/data/output_json"
BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
base_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs"

def extract_stroke_batsman_gt(vidsPath, labelsPath, batPath, partition_lst):
    """
    Function to iterate on all the training videos and extract the relevant features.
    vidsPath: str
        path to the dataset containing the videos
    labelsPath: str
        path to the JSON files for the labels.
    batPath : str
        path to the txt files for the batsman pose labels (manual annotations)
    partition_lst: list of video_ids
        video_ids are the filenames (without extension)
    """
    mth, nbins = 0, 20
    strokes_name_id = []
    all_feats = {}
    bins = np.linspace(0, 2*np.pi, (nbins+1))
        
    for i, v_file in enumerate(partition_lst):
        print('-'*60)
        print(str(i+1)+". v_file :: ", v_file)
        if '.avi' in v_file or '.mp4' in v_file:
            v_file = v_file.rsplit('.', 1)[0]
        json_file = v_file + '.json'
        bat_gt_file = v_file + '_gt.txt'
        
        # read labels from JSON file
        assert os.path.exists(os.path.join(labelsPath, json_file)), "{} doesn't exist!".format(json_file)
        assert os.path.exists(os.path.join(batPath, bat_gt_file)), "{} doesn't exist!".format(bat_gt_file)
        
        with open(os.path.join(labelsPath, json_file), 'r') as fr:
            frame_dict = json.load(fr)
        frame_indx = list(frame_dict.values())[0]                
        
        with open(os.path.join(batPath, bat_gt_file), "r") as fp:
            gt_rows = fp.readlines()
            
        # Create VideoPose object
        vid_pose = VideoPoseTrack(DATASET, v_file+".avi", POSE_FEATS, 
                                  os.path.join(labelsPath, json_file))
        
        # # remove \n at end and split into list of tuples
        # eg. tuple is ['98', '1', '303', '28', '353', '130', 'Batsman']
        gt_rows = [line.strip().split(',') for line in gt_rows]
#        gt_rows.reverse()
        
        for m,n in frame_indx:
            k = v_file+"_"+str(m)+"_"+str(n)
            print("Stroke {} - {}".format(m,n))
            strokes_name_id.append(k)
            # Extract the stroke features
            # Extract bboxes corresponding to batsman only
#            all_feats[k] = extract_stroke_bboxes(os.path.join(vidsPath, v_file+".avi"), m, n, gt_rows)
#            # Extract flow_angles (HOOF) for areas contained inside batsman bboxes
#            all_feats[k] = extract_flow_angles_masked(os.path.join(vidsPath, v_file+".avi"), 
#                             m, n, gt_rows, bins, mth, True)
#            # Extract pose keypoint features extracted from OPENPOSE
            all_feats[k] = extract_stroke_poses(os.path.join(vidsPath, v_file+".avi"), 
                                                m, n, gt_rows, vid_pose)
        #break
    return all_feats, strokes_name_id

def extract_stroke_bboxes(vidFile, start, end, gt_rows):
    '''Find the rows belonging to stroke and for a matrix of bbox entries.
    Returns:
        np.array for bboxes 
    '''
    boxes = get_bboxes_for_stroke(gt_rows, start, end)
    return np.array(boxes)

def extract_flow_angles_masked(vidFile, start, end, gt_rows, hist_bins, mag_thresh, density=False):
    '''
    Extract optical flow maps from video vidFile for all the frames and put the angles with >mag_threshold in different 
    bins. The bins vector is the feature representation for the stroke. 
    Use only the strokes given by list of tuples frame_indx.
    Parameters:
    ------
    vidFile: str
        complete path to a video
    start: int
        starting frame number
    end: int
        ending frame number
    gt_rows : list of tuples
        annotations for the batsman bboxes. Contains all the annotations for the video
    hist_bins: 1d np array 
        bin divisions (boundary values). Used np.linspace(0, 2*PI, 11) for 10 bins
    mag_thresh: int
        minimum size of the magnitude vectors that are considered (no. of pixels shifted in consecutive frames of OF)
    
    '''
    boxes = get_bboxes_for_stroke(gt_rows, start, end)
    boxes.reverse()
    cap = cv2.VideoCapture(vidFile)
    if not cap.isOpened():
        print("Capture object not opened. Aborting !!")
        sys.exit(0)
    ret = True
    W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    stroke_features = []
    prvs, next_ = None, None
    m, n = start, end    
    #print("stroke {} ".format((m, n)))
    sum_norm_mag_ang = np.zeros((len(hist_bins)-1))  # for optical flow maxFrames - 1 size
    frameNo = m
    while ret and frameNo <= n:
        if (frameNo-m) == 0:    # first frame condition
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNo)
            ret, frame1 = cap.read()
            if not ret:
                print("Frame not read. Aborting !!")
                break
            # resize and then convert to grayscale
            #cv2.imwrite(os.path.join(flow_numpy_path, str(frameNo)+".png"), frame1)
            prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            #prvs = scale_and_crop(prvs, scale)
            frameNo +=1
            continue
            
        ret, frame2 = cap.read()
        # resize and then convert to grayscale
        next_ = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        if len(boxes) > 0:
            x0, y0, x1, y1 = boxes.pop()
            x0 = max(0, x0 - 10)
            y0 = max(0, y0 - 10)
            x1 = min(W, x1 + 10)
            y1 = min(H, y1 + 10)
        else:
            break
        
        flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        
        #print("Mag > 5 = {}".format(np.sum(mag>THRESH)))
        pixAboveThresh = np.sum(mag>mag_thresh)
        #use weights=mag[mag>THRESH] to be weighted with magnitudes
        #returns a tuple of (histogram, bin_boundaries)
        mag, ang = mag[y0:(y1+1), x0:(x1+1)], ang[y0:(y1+1), x0:(x1+1)]
        ang_hist = np.histogram(ang, bins=hist_bins, density=density)
        stroke_features.append(ang_hist[0])
        #sum_norm_mag_ang +=ang_hist[0]
#            if not pixAboveThresh==0:
#                sum_norm_mag_ang[frameNo-m-1] = np.sum(mag[mag > THRESH])/pixAboveThresh
#                sum_norm_mag_ang[(maxFrames-1)+frameNo-m-1] = np.sum(ang[mag > THRESH])/pixAboveThresh
        frameNo+=1
        prvs = next_
        #stroke_features.append(sum_norm_mag_ang/(n-m+1))
    cap.release()
    #cv2.destroyAllWindows()
    stroke_features = np.array(stroke_features)
    #Normalize row - wise
    #stroke_features = stroke_features/(1+stroke_features.sum(axis=1)[:, None])
    return stroke_features


def extract_stroke_poses(vidFile, start, end, gt_rows, vid_pose, density=False):
    '''
    Extract player pose keypoints for the stroke frames using VideoPoseTrack object 
    
    Parameters:
    ------
    vidFile: str
        complete path to a video
    start: int
        starting frame number
    end: int
        ending frame number
    gt_rows : list of tuples
        annotations for the batsman bboxes. Contains all the annotations for the video
    vid_pose: VideoPoseTrack obj
        VideoPoseTrack object corresponding to the highlights video file containing 
        the stroke.
    '''
    print("Video Pose kp : ")
    vidPoseVecs = vid_pose.getPoseFeatures()
    gt_bboxes = get_bboxes_for_stroke(gt_rows, start, end)
    batsmanPoses = []
    # consider poses for stroke temporal location
    for i, pos in enumerate(list(range(start, end+1))):
#        strokePoseVecs.append(vidPoseVecs[i])
        if i < len(gt_bboxes):
            batsmanPoses.append(findBatsmanPose(vidPoseVecs[pos], gt_bboxes[i]))
        else:
            break
    # remove None values from list, occurs when strokePoseVecs is empty. Wrong openpose kp.
    batsmanPoses = [p for p in batsmanPoses if p is not None]
    # remove probabilities of keypoints, keep only the spatial locations
    batsmanPoses = np.array(batsmanPoses)
    assert batsmanPoses.shape[1] == 75, "Invalid shape of the poses matrix!"
    # Get 25 (x,y) keypoints. Deleting probability values associated to keypoints
    batsmanPoses = np.delete(batsmanPoses, list(range(2, batsmanPoses.shape[1], 3)), axis=1)
    # Fill 0 values with centroids of mean of the coordinate corresponding to the pose
    # fill with nan
    batsmanPoses[batsmanPoses == 0] = np.nan    
    xPoses = batsmanPoses[:, range(0, 50, 2)]
    yPoses = batsmanPoses[:, range(1, 50, 2)]

    xMean, yMean = np.nanmean(xPoses, 1), np.nanmean(yPoses, 1)
    #Find indices that you need to replace
    x_inds, y_inds = np.where(np.isnan(xPoses)), np.where(np.isnan(yPoses))
    #Place row means in the indices. Align the arrays using take
    xPoses[x_inds] = np.take(xMean, x_inds[0])
    yPoses[y_inds] = np.take(yMean, y_inds[0])
    
    return np.hstack((xPoses, yPoses))
    
def findBatsmanPose(strokePoseVecs, gt_bbox):
    '''Find strokePose having max IOU with ground truth bounding box.
    '''
    frameBBoxes = utils.convertPoseVecsToBBoxes(strokePoseVecs)
    iouWithPoses = [getIOU_poseWithBB(pose_bb, gt_bbox) for pose_bb in frameBBoxes]
    # check if all the iou values are 0, or size of iouWithPoses is 0
    if all(p==0 for p in iouWithPoses): # Covers the case of len(iouWithPoses) == 0 :
        return None
    
    batsman_pose_idx = iouWithPoses.index(max(iouWithPoses))
    return strokePoseVecs[batsman_pose_idx]
    

def get_closest_pose():
    '''If all the IOU values are 0, then find use the pose which is closest to the 
    GT BBox. 
    '''
    return 0
    
def getIOU_poseWithBB(pose_bb, gt_bb):
    '''
    '''
    x00, y00, x01, y01 = pose_bb
    x10, y10, x11, y11 = gt_bb
    overlap_xlen =  min(x01, x11) - max(x00, x10)
    overlap_ylen =  min(y01, y11) - max(y00, y10)
    
    if overlap_xlen > 0 and overlap_ylen > 0:
        intersect_area = overlap_xlen * overlap_ylen
    else:
        return 0.0
    
    # compute the area of both BBs
    bb1_area = (x01 - x00) * (y01 - y00)
    bb2_area = (x11 - x10) * (y11 - y10)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersect_area / float(bb1_area + bb2_area - intersect_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_bboxes_for_stroke(gt_rows, start, end):
    '''Return the boxes for the stroke starting from 'start' to 'end'
    Parameters:
    -----------
    start: int
        starting frame number
    end: int
        ending frame number
    gt_rows : list of tuples
        annotations for the batsman bboxes. Contains all the annotations for the video
        
    Returns: 
    --------
    list of tuples [[x0, y0, x1, y1], ....]  
    '''            
    boxes = []
    for tup in gt_rows:
        fno, lab_id, x0, y0, x1, y1, label = tup
        if int(fno) >= start and int(fno) <= end and label=='Batsman':
            boxes.append([int(x0), int(y0), int(x1), int(y1)])
            
    return boxes

if __name__ == '__main__':

    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
    base_name = os.path.join(base_path, "bow_HL_batsman_poses_fillNA")
    if not os.path.isdir(base_name):
        os.makedirs(base_name)
    ###########################################################################
    # 1: Extraction of batsman poses and visualize on the frames and plots
    features, strokes_name_id = extract_stroke_batsman_gt(DATASET, LABELS, BAT_LABELS, train_lst)

    with open(os.path.join(base_name, "kp_gt_feats.pkl"), "wb") as fp:
        pickle.dump(features, fp)
    with open(os.path.join(base_name, "kp_gt_snames.pkl"), "wb") as fp:
        pickle.dump(strokes_name_id, fp)
    
    # For validation set
    features_val, strokes_name_id_val = extract_stroke_batsman_gt(DATASET, LABELS, BAT_LABELS, val_lst)

    with open(os.path.join(base_name, "kp_gt_feats_val.pkl"), "wb") as fp:
        pickle.dump(features_val, fp)
    with open(os.path.join(base_name, "kp_gt_snames_val.pkl"), "wb") as fp:
        pickle.dump(strokes_name_id_val, fp)

    features_test, strokes_name_id_test = extract_stroke_batsman_gt(DATASET, LABELS, BAT_LABELS, test_lst)

    with open(os.path.join(base_name, "kp_gt_feats_test.pkl"), "wb") as fp:
        pickle.dump(features_test, fp)
    with open(os.path.join(base_name, "kp_gt_snames_test.pkl"), "wb") as fp:
        pickle.dump(strokes_name_id_test, fp)
        
        

#    ###########################################################################
    # get list of label filenames containing temporal stroke and batsman labels
#    train_lab = [f.rsplit('.',1)[0] +".json" for f in train_lst]
#    val_lab = [f.rsplit('.',1)[0] +".json" for f in val_lst]
#    test_lab = [f.rsplit('.',1)[0] +".json" for f in test_lst]
#    train_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in train_lst]
#    val_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in val_lst]
#    test_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in test_lst]
#    
#    ###########################################################################
#    
#    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
#    val_labs = [os.path.join(LABELS, f) for f in val_lab]
#    tr_gt_batsman = [os.path.join(BAT_LABELS, f) for f in train_gt_batsman]
#    sizes = [utils.getTotalFramesVid(os.path.join(DATASET, f)) for f in train_lst]
#    print("Size : {}".format(sizes))
    
#    bboxes = True
#    epsilon = 50
#    visualize = False
#    n_clusters = 5
#    clust_no = 1
#    first_frames = 2

#    ###########################################################################
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
    
#    # 2: Track the poses for strokes
#    track_lst, count = track_stroke_poses(train_lst, tr_labs, tr_gt_batsman, epsilon, bboxes, visualize)
#    
#    #plot_tracks(bboxes)
    
    
#def get_stacked_poses_df(train_lst, tr_labs, tr_gt_batsman, bboxes=True, n=1):
#    '''
#    Create a dataframe of the first n frame poses with "vidname_frameNo_P0" as rownames.
#    Total number of rows are \Sigma_{i} \Sigma_{j} {P_{ij}} where j is pose in each frame
#    and i iterates on selected frames. For n=1 i iterates on only first frames of each
#    stroke.
#    
#    Parameters:
#    ----------
#    train_lst : list of str
#        list of video file names for which poses have to be extracted
#    tr_labs : list of str
#        list of paths of corresponding json files having stroke labels
#    tr_gt_batsman : list of str
#        list of paths of corresponding txt files having batsman locations
#    bboxes : boolean
#        True if bounding boxes have to be extracted, False if 75 dimensional 
#        pose vectors have to be extracted
#    n : int
#        No. of frames as starting of the stroke that have to be considered for 
#        extraction
#        
#    Returns:
#    -------
#    pd.DataFrame : an (P x 75) or (P x 4) dataframe with rownames as 
#    vidname_frmNo_PoseNo where P is total number of poses in set of frames
#    
#    '''
#    poses_lst = []
#    batsman_poses_gt = []
#    for i, vid_file in enumerate(train_lst):
#        # Create object for one video only
#        v = VideoPoseTrack(DATASET, train_lst[i], POSE_FEATS, tr_labs[i])
#        
#        # get list of pose matrices for target frames by iterating on strokes
#        # each sublist is for a stroke with n arrays for n first frames
#        topn_poses = v.getCAMFramePoses(first_n = n, bounding_box=bboxes)
#        batsman_poses_gt.append({vid_file : utils.getGTBatsmanPoses(BAT_LABELS, \
#                                                            tr_gt_batsman[i])})
#        #v.visualizeVideoWithBatsman()
#        # Iterate on the strokes to get the frameNo and PoseNo for rownames
#        for stroke_idx, (beg, end) in enumerate(v.strokes):
#            for frm_no in range(n):     # Iterate over first n frames of each stroke
#                # get the matrix of poses for a selected frame
#                pose_arr = topn_poses[stroke_idx][frm_no]
#                rownames = []
#                for pose_no in range(pose_arr.shape[0]):    # Iterate on the poses
#                    rownames.append(train_lst[i]+"_"+str(beg+frm_no)+"_P"+str(pose_no))
#                # create a sub-dataframe and append to list
#                poses_lst.append(pd.DataFrame(pose_arr, index=rownames))
#    return pd.concat(poses_lst), batsman_poses_gt     # vstack all sub-dataframes
#    
    
