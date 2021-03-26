#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:11:34 2019

@author: arpan

@Description: Utils file for tracking batsman's pose.
"""

import json
import os
import cv2
import time
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

_WHITE = (255, 255, 255)
_GREEN = (18, 127, 15)
_RED = (240, 15, 15)

person_colors = [[240,  248,  255], # aliceblue
              [0,    0,    0 ],  # black
              [0,    0,    255], # blue
              [165,  42,   42 ], # brown
              [220,  20,   60 ], # crimson
              [0,    100,  0  ], # dark green
              [148,  0,    211], # dark violet
              [173,  255,  47],  # green yellow
              [173,  216,  230], # light blue
              [255,  0,    255], # magenta
              [255,  165,  0  ], # orange
              [255,  0,    0  ], # red
              [255,  255,  255], # white
              [255,  255,  0  ], # yellow
              [154,  205,  50] ]   # yellow green
              
#person_colors=[[255,255,255],[255,0,0],[0,255,0],[0,0,255],[255,255,0], 
#                   [0,255,255],[255,0,255],[0,128,128],[0,0,0]]
NCOLORS = len(person_colors)


def split_dataset_files(datasetPath):
    '''Split the dataset files into training, validation and test sets.
    Only for the highlights dataset. It is assumed that all the video files 
    are at the same path. 
    Parameters:
    ----------
    datasetPath : str
        complete path to the video dataset. The folder contains 26 video files
        
    Returns:
    -------
    filenames : list of str
        3 lists with filenames. 16 for training, 5 for validation and 5 for testing
    '''
    filenames = sorted(os.listdir(datasetPath))         # read the filename
#    filenames = [t.split('.')[0] for t in filenames]   # remove the extension
    return filenames[:16], filenames[16:21], filenames[21:]


def getImageFromName(datasetPath, pose_fname):
    '''Function to get the image from the video for the corresponding json file
    Parameters:
    -----------
    datasetPath : str
        path to the video dataset
    pose_fname : str
        filename of the pose json file, like <vidname>_000000000234_keypoints.json
    
    Returns:
    --------
    img : a (c, H, W) numpy matrix containing the frame 234 from <vidname>
    '''
    # get a list with 0 index as vName and 1 index as Frame No.
    vNameFrame = pose_fname.rsplit('_', 2)
    vName = vNameFrame[0]
    vFrameNo = int(vNameFrame[1])
    cap = cv2.VideoCapture(os.path.join(datasetPath, vName))
    if not cap.isOpened():
        print("Capture object not opened !! Abort !")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, vFrameNo)
    ret, img = cap.read()
    if not ret:
        print("Frame not read !!")
        return 
#    plt.figure()
#    plt.imshow(img)
    cap.release()
    return img

def getTotalFramesVid(srcVideoPath):
    """
    Return the total number of frames in the video
    
    Parameters:
    ------
    srcVideoPath: str
        complete path of the source input video file
        
    Returns:
    ------
    total frames present in the given video file
    """
    cap = cv2.VideoCapture(srcVideoPath)
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return 0

    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int(tot_frames)

def getGTBatsmanPoses(gt_path, gtFile, str_label='Batsman'):
    """
    Return the list of pose bounding box vectors for the batsman. 
    For frames in trimmed videos, get vectors shape (4,), 
    where 4 is the sequence of (x0, y0, x1, y1) coordinates of batsman.
    Parameters:
    -----------
    gt_path: str
        path with csv files of batsman labels
    gtFile: str
        filename for batsman labels for current video
    str_label: str
        boxes read for this actor poses
        
    Returns: 
    --------
    dict like {FrameNo : [x0, y0, x1, y1], ...} format corresponding to str_label
    """
    gt_batsman = {}
    assert os.path.exists(os.path.join(gt_path, gtFile)), \
                            "{} does not exist".format(gtFile)
    import csv
    with open(os.path.join(gt_path, gtFile), 'r') as fp:
        data = csv.reader(fp)
        for row in data:
            if type(row)==list and len(row)>0 and row[-1]==str_label:
                # save in {FrameNo : [x0, y0, x1, y1]} format
                gt_batsman[int(row[0])] = [int(row[2]), int(row[3]), \
                                int(row[4]), int(row[5])]
    return gt_batsman

def get_avg_box_values(batsman_gt_train):
    '''Average out the box dimensions of the batsman poses and return.
    Parameters:
    -----------
    batsman_gt_train : list of dictionaries
        list of dictionary values as [{vid1 : {98: [x0, y0, x1, y1], 
        99: [x0, y0, x1, y1], ...}} , {vid2: { ... }} , ... ] where 98, 99 are 
        frame numbers as keys with box coordinates are values.
    
    Returns:
    --------
    nboxes : Total number of boxes
    tot_width/nboxes : Averaged width of boxes
    tot_height/nboxes : Averaged height of boxes
    area/nboxes : Averaged area of the boxes
    '''
    # Find mean width and mean height of bounding boxes
    nboxes, tot_width, tot_height, area = 0, 0, 0, 0
    for vid_gt in batsman_gt_train:
        # get video name as key value
        vidname = list(vid_gt.keys())[0]
        for k, box in vid_gt[vidname].items():
            tot_width += (box[2] - box[0])
            tot_height += (box[3] - box[1])
            area += ((box[2] - box[0]) * (box[3] - box[1]))
            nboxes+=1
    if nboxes == 0:
        return 0, 0, 0, 0
    return nboxes, tot_width/nboxes, tot_height/nboxes, area/nboxes

def plotPersonFromMatrix(img, poses, people_ordering):
    '''
    img: input image RGB
    pose_kp: 2D matrix of N_persons X 75 .
    '''
    
    nPeople = poses.shape[0]
    for pid in range(nPeople):
        person_pose_vals = poses[pid, :]    # get vector of shape (75,)
        s=people_ordering[pid]
        p_col = person_colors[s % NCOLORS]
        for j in range(0, len(person_pose_vals), 3):
            x = int(person_pose_vals[j])
            y = int(person_pose_vals[j+1])
            c = person_pose_vals[j+2]
            
            img[(y-2):(y+2),(x-2):(x+2),:] = p_col  # display color at keypoints
        #break  # uncomment to display for only one person in the image
#    plt.figure()
#    plt.imshow(img)
    return 

def plotPersonBoundingBoxes(img, poses_bboxes, people_ordering, box_thickness=1):
    '''
    img: input image RGB
    poses: 2D matrix of bounding boxes of size N_persons X 4 .
    people_ordering: sequence of non-negative integers denoting N_persons
    '''
    nPeople = poses_bboxes.shape[0]
    for pid in range(nPeople):
        person_pose = poses_bboxes[pid, :]    # get vector of shape (75,) or (4,)
        colorId = people_ordering[pid]
        p_col = person_colors[colorId % NCOLORS]  # get color tuple
        x0, y0 = person_pose[0], person_pose[1]
        x1, y1 = person_pose[2], person_pose[3]
        #print("Person : {} x:".format(pid), x_pose_points)
        #print("Person : {} y:".format(pid), y_pose_points)
        cv2.rectangle(img, (x0, y0), (x1, y1), p_col, thickness=box_thickness)

    
def highlight_selected_box(img, pose_mat, people_ordering, refpt):
    
    if pose_mat.shape[0] != 4:
        pose_mat = convertPoseVecsToBBoxes(pose_mat)
    for pid in range(pose_mat.shape[0]):
        person_pose = pose_mat[pid, :]    # get vector of shape (75,)
        s = people_ordering[pid]
        p_col = person_colors[s % NCOLORS]
        x0, y0 = int(person_pose[0]), int(person_pose[1])
        x1, y1 = int(person_pose[2]), int(person_pose[3])
        if (refpt[0]>=x0 and refpt[0]<=x1) and (refpt[1]>=y0 and refpt[1]<=y1):
            cv2.rectangle(img, (x0, y0), (x1, y1), p_col, thickness=3)
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), p_col, thickness=1)
    return 

def getPoseVector(pose_kp):
    """
    Receives the pose dictionary and returns the person keypoints as a matrix.
    Returns : n_persons_in_frame x 75 sized np.ndarray
    """
    people = pose_kp['people']
    people_list = []
    for i, p in enumerate(people):
        # get list of values (x1, y1 ,c1, x2, y2, c2 ...)
        person_pose_vals = p['pose_keypoints_2d']   
        people_list.append(person_pose_vals)
    return np.array(people_list)      # return 2D matrix persons 
    
def convertPoseVecsToBBoxes(person_kp_mat):
    """
    Receives the person pose matrix and return the person bboxes as a matrix of size
    (N_Persons X 4)
    
    Parameters:
    -----------
    person_kp_mat: np.array shape : (N_person_in_frame, 75)
        contains the pose vectors of persons as row vectors in the matrix
    
    Returns:
    --------
    np.array shape : (N_person_in_frame, 4) : N bounding boxes for N persons
    """
    people_list = []
    # iterate on the rows, each row representing single person vector of size (75,)
    for i, person_pose in enumerate(person_kp_mat):
        # get list of values (x1, y1 ,c1, x2, y2, c2 ...)
        x_pose_points = [int(person_pose[j]) for j in range(0, len(person_pose), 3) \
                             if int(person_pose[j])!=0]
        y_pose_points = [int(person_pose[j]) for j in range(1, len(person_pose), 3) \
                             if int(person_pose[j])!=0]
        x0, y0 = min(x_pose_points), min(y_pose_points)
        x1, y1 = max(x_pose_points), max(y_pose_points)
        
        people_list.append([x0, y0, x1, y1])
        
    return np.array(people_list)      # return 2D matrix persons BBoxes

def get_next_mapping(p1, p2, pid1, next_pid=0, epsilon=20):
    '''Find the next mapping of poses in current frame corresponding to the previous 
    frame mapping.
    Parameters:
    ----------
    
    '''
    # find pairwise euclidean distances between the poses of two consecutive frames
    if p1.shape[0] == 0:
        if p2.shape[0] == 0:
            return []
        else:
            return list(range(next_pid, next_pid + p2.shape[0]))
    else:
        if p2.shape[0] == 0:
            return []
    
    # only calculated when both p1 and p2 have rows
    dist = euclidean_distances(p1, p2)
    
    # get closest poseNo ordering according to min euclidean distances (row-wise)
    closest_poses = np.argmin(dist, axis = 1)
    
    # Initialize a vector of poseNos for next frame
    pid2 = -np.ones(p2.shape[0], dtype='int')
    
    # Assign the poseNos from previous frame to curr frame
    # Iterate over each person in p1
    for i, pid1_i in enumerate(pid1):
        # goto row and find min distance column number
        #print("distance : {}".format(dist[i, closest_poses[i]]))
        if dist[i, closest_poses[i]] <= epsilon:
            pid2[closest_poses[i]] = pid1_i
    
    # if increase/decrease in number of poses
    # used if curr frame has more poses than previous frame
    for idx in np.where(pid2==-1)[0]:
        pid2[idx] = next_pid
        next_pid +=1
        
    return list(pid2)

def get_track2frame(stroke_tracks):
    '''Receive a list of lists corresponding to the poses in each frame of a 
    stroke and convert to a track_id/pose_id to frame no list. The length of 
    the list gives the length of the pose track.
    Parameters:
    -----------
    stroke_tracks : list of list
        The sublists contain pose ids in each frame of the stroke.
        
    Returns:
    --------
    dict with key as pose_ids and values as list of frames_nos with tracks
    eg., {pose_id1 : [f1, f2, f3, ...], ...} 
    
    '''
    track2frame = {}
    for frm_no, frm_poses in  enumerate(stroke_tracks):
        for pid in frm_poses:
            if pid not in track2frame:
                track2frame[pid] = [frm_no]
            else:
                track2frame[pid].append(frm_no)
    return track2frame

def normalize_pose(p):
    '''
    '''
    
    if p.shape[-1] == 0:
        return p
    
    shift = 2
    if p.shape[-1] == 75:
        shift = 3
    
    # operate on only x coordinate values
    p[:, ::shift] = p[:, ::shift] - np.min(p[:, ::shift], axis=1)[:, None]
    # operate on only y corrdinate values
    p[:, 1::shift] = p[:, 1::shift] - np.min(p[:, 1::shift], axis=1)[:, None]
    # calculate box widths and heights
    bb_wd = (np.max(p[:, ::shift], axis=1) - np.min(p[:, ::shift], axis=1))[:, None]
    bb_ht = (np.max(p[:, 1::shift], axis=1) - np.min(p[:, 1::shift], axis=1))[:, None]
    # divide by box widths and heights
    p[:, ::shift] = p[:, ::shift] / bb_wd
    p[:, ::shift] = p[:, 1::shift] / bb_ht
    
    return p
        
def waitTillEscPressed():
    '''
    Supporting method for visualizing output frame by frame, continuing as keystrokes
    '''
    while(True):
        # For moving forward
        if cv2.waitKey(0)==27:
            print("Esc Pressed. Move Forward.")
            return 1
        # For moving back
        elif cv2.waitKey(0)==98:
            print("'b' pressed. Move Back.")
            return 0
        # move to next shot segment
        elif cv2.waitKey(0)==110:
            print("'n' pressed. Move to next shot.")
            return 2
        # move to next video
        elif cv2.waitKey(0)==112:
            print("'p' pressed. Move to next video.")
            return 3

def collate_fn(batch):
    return tuple(zip(*batch))

#def showPersonBBox(img, bbox, thick=1):
#    """
#    Taken from detectron vis_bbox in utils/vis.py 
#    """
#    (x0, y0, w, h) = bbox
#    x1, y1 = int(x0+w), int(y0+h)
#    x0, y0 = int(x0), int(y0)
#    cv2.rectangle(img, (x0, y0), (x1, y1), _RED, thickness=thick)
#    return img
    
#def mapping(pose_img1, pose_img2, arr):
#        a=len(pose_img1['people'])
#        b=len(pose_img2['people'])
#        print(a)
#        print(b)
#        arr_map=[]
#        
#        for p in range(b):
#            mini=10**1000
#            
#            for q in range(a):
#                sums=0
#                
#                for r in range(0,75,3):
#                    x2=pose_img2['people'][p]['pose_keypoints_2d'][r]
#                    y2=pose_img2['people'][p]['pose_keypoints_2d'][r+1]
#                    x1=pose_img1['people'][q]['pose_keypoints_2d'][r]
#                    y1=pose_img1['people'][q]['pose_keypoints_2d'][r+1]
#                    p1=np.array((x1,y1))
#                    p2=np.array((x2,y2))
#                    dist= np.linalg.norm(p2-p1)
#                    sums= np.add(sums,dist)
#
#                if(min(mini,sums)==sums):
#                    mini=sums
#                    s=q
#            m=arr[s]
#            arr_map.append(m)
#            
#        return arr_map
#
#def plotPerson(img, pose_kp, people_ordering):
#    '''
#    img: input image RGB
#    pose_kp: dictionary formed by openpose for the corresponding image
#    '''
#    people = pose_kp['people']
#    people_ordering_new = []
#    for o,p in enumerate(people):
#        # get list of values (x1, y1 ,c1, x2, y2, c2 ...)
#        person_pose_vals = p['pose_keypoints_2d']   
#        s=people_ordering[o]
#        people_ordering_new.append(s)
#        p_col = person_colors[s%NCOLORS]
#        for j in range(0, len(person_pose_vals), 3):
#            x = int(person_pose_vals[j])
#            y = int(person_pose_vals[j+1])
#            c = person_pose_vals[j+2]
#            
#            img[(y-2):(y+2),(x-2):(x+2),:] = p_col  # do white
#        
#        #break  # remove to display for all the persons in the image
#        
##    plt.figure()
##    plt.imshow(img)
#    #return img
#    return people_ordering_new