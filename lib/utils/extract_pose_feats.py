#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 01:05:08 2020

@author: arpan

@Description: Extract Pose Based features from videos serially/parallely.
"""

import _init_paths

import os
import cv2
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from utils import track_utils as utils

_WHITE = (255, 255, 255)
_GREEN = (18, 127, 15)
_RED = (240, 15, 15)

def extract_all(srcFolderPath, destFolderPath, onGPU=True, stop='all'):
    """
    Function to extract the features from a list of videos
    
    Parameters:
    ------
    destFolderPath: str
        path to store the optical flow values in .bin files
    onGPU: boolean
        True enables a serial extraction by sending model and data to GPU,
        False enables a parallel extraction on the different CPU cores.
    stop: str
        to traversel 'stop' no of files in each subdirectory.
    
    Returns: 
    ------
    traversed: int
        no of videos traversed successfully
    """
    # iterate over the subfolders in srcFolderPath and extract for each video 
    vfiles = os.listdir(srcFolderPath)
    
    infiles, outfiles, nFrames = [], [], []
    
    traversed = 0
    # create destination path to store the files
    if not os.path.exists(destFolderPath):
        os.makedirs(destFolderPath)
            
    # iterate over the video files inside the directory sf
    for vid in vfiles:
        if os.path.isfile(os.path.join(srcFolderPath, vid)) and \
                                    vid.rsplit('.', 1)[1] in {'avi', 'mp4'}:
            infiles.append(os.path.join(srcFolderPath, vid))
            outfiles.append(os.path.join(destFolderPath, vid.rsplit('.', 1)[0]+".npy"))
            nFrames.append(utils.getTotalFramesVid(os.path.join(srcFolderPath, vid)))
            # save at the destination, if extracted successfully
            traversed += 1
#            print "Done "+str(traversed_tot+traversed)+" : "+sf+"/"+vid
                    
                # to stop after successful traversal of 2 videos, if stop != 'all'
            if stop != 'all' and traversed == stop:
                break
                    
    print("No. of files to be written to destination : "+str(traversed))
    if traversed == 0:
        print( "Check the structure of the dataset folders !!")
        return traversed
    ###########################################################################
    #### Form the pandas Dataframe and parallelize over the files.
    filenames_df = pd.DataFrame({"infiles":infiles, "outfiles": outfiles, "nframes": nFrames})
    filenames_df = filenames_df.sort_values(["nframes"], ascending=[True])
    filenames_df = filenames_df.reset_index(drop=True)
    nrows = filenames_df.shape[0]
    batch = 2  # No. of videos in a single batch
    njobs = 1   # No. of threads
    
    ###########################################################################
    if onGPU:
        # Serial Implementation (For GPU based extraction)
        for i in range(nrows):
            st = time.time()
            feat = getPoseFeats(filenames_df['infiles'][i], onGPU)
            # save the feature to disk
            if feat is not None:
                np.save(filenames_df['outfiles'][i], feat)
                print("Written "+str(i)+" : "+filenames_df['outfiles'][i])
                
            e = time.time()
            print( "Execution Time : "+str(e-st))
    
    else:    
        #feat = getC3DFrameFeats(model, filenames_df['infiles'][0], onGPU, depth)
        # Parallel version (For CPU based extraction)
        for i in range(nrows/batch):
            batch_diffs = Parallel(n_jobs=njobs)(delayed(getPoseFeats) \
                        (filenames_df['infiles'][i*batch+j], onGPU) \
                        for j in range(batch))
            print("i = "+str(i))
            # Writing the diffs in a serial manner
            for j in range(batch):
                if batch_diffs[j] is not None:
                    np.save(filenames_df['outfiles'][i*batch+j], batch_diffs[j])
                    print("Written "+str(i*batch+j+1)+" : "+ \
                                filenames_df['outfiles'][i*batch+j])
                
        # For last batch which may not be complete, extract serially
        last_batch_size = nrows - ((nrows/batch)*batch)
        if last_batch_size > 0:
            batch_diffs = Parallel(n_jobs=njobs)(delayed(getPoseFeats) \
                        (filenames_df['infiles'][(nrows/batch)*batch+j], onGPU) \
                        for j in range(last_batch_size)) 
            # Writing the diffs in a serial manner
            for j in range(last_batch_size):
                if batch_diffs[j] is not None:
                    np.save(filenames_df['outfiles'][(nrows/batch)*batch+j], batch_diffs[j])
                    print("Written "+str((nrows/batch)*batch+j+1)+" : "+ \
                                filenames_df['outfiles'][(nrows/batch)*batch+j])
    
    ###########################################################################
#    print len(batch_diffs)
    return traversed



def getPoseFeats(DATASET, POSE_FEATS, videoName, onGPU):
    """
    Function to read all the frames of the video and get sequence of features
     one batch at a time. 
    This function can be called parallely based on the amount of available
    memory.
    
    Parameters:
    ------
    videoName: str
        file name of the video.
        
    Returns:
    ------
    np.array of size (N-depth+1) x 4096 (N is the no. of frames in video.)
    """
    
    # get the VideoCapture object
    cap = cv2.VideoCapture(os.path.join(DATASET, videoName))
    prefix = videoName.rsplit('.', 1)[0]
    
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return None
    
    W, H = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frameCount = 0
    features_current_file = []
    
    #ret, prev_frame = cap.read()
    assert cap.isOpened(), "Capture object does not return a frame!"
    #prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    X = []  # input, initially a list, after first 16 frames converted to ndarray
    # Iterate over the entire video to get the optical flow features.
    while(cap.isOpened()):
        
        ret, curr_frame = cap.read()    # H x W x C
        if not ret:
            break
        poseFile = os.path.join(POSE_FEATS, prefix+'_{:012}'.format(frameCount)+\
                                '_keypoints.json')
        if not os.path.exists(poseFile):
            print("File does not exist !! {}".format(frameCount))
            frameCount +=1
            continue
            
        # resize to 180 X 320 and taking centre crop of 112 x 112
        #curr_frame = cv2.resize(curr_frame, (W/2, H/2), cv2.INTER_AREA)
        #(h, w) = curr_frame.shape[:2]
        # size is 112 x 112 x 3
        #curr_frame = curr_frame[(h/2-56):(h/2+56), (w/2-56):(w/2+56), :]
        with open(poseFile, 'r') as fp:
            pose_dict = json.load(fp)
        curr_frame = utils.plotPoseKeyPoints(curr_frame, pose_dict)
        #print(type(curr_frame), curr_frame.shape)
        
        if frameCount % 100 == 0:
            plt.figure()
            plt.imshow(curr_frame)
        
        frameCount +=1
        #print "{} / {}".format(frameCount, totalFrames)

    # When everything done, release the capture
    cap.release()
    #return features_current_file
    #return np.array(features_current_file)      # convert to (N-depth+1) x 1 x 4096
