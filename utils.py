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
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# Split the dataset files into training, validation and test sets
# All video files present at the same path (assumed)
def split_dataset_files(datasetPath):
    filenames = sorted(os.listdir(datasetPath))         # read the filename
    filenames = [t.split('.')[0] for t in filenames]   # remove the extension
    return filenames[:16], filenames[16:21], filenames[21:]


# function to get the image from the video for the corresponding json file
def getImageFromName(DATASET, pose_fname):
    vNameFrame = pose_fname.rsplit('.',1)[0].rsplit('_', 2)   # get a list with 0 index as vName and 1 index as Frame No.
    vName = vNameFrame[0]+'.avi'
    vFrameNo = int(vNameFrame[1])
    cap = cv2.VideoCapture(os.path.join(DATASET, vName))
    if not cap.isOpened():
        print("Capture object not opened !! Abort !")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, vFrameNo)
    ret, img = cap.read()
    if not ret:
        print("Frame not read !!")
        return 
#    
#    plt.figure()
#    plt.imshow(img)
    cap.release()
    return img

        
def plotPerson(img, pose_kp, people_ordering):
    colorlist=[[255,255,255],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[0,128,128],[0,0,0]]
    '''
    img: input image RGB
    pose_kp: dictionary formed by openpose for the corresponding image
    '''
    people = pose_kp['people']
    nPeople = len(people)
    people_ordering_new = []
    for o,p in enumerate(people):
        person_pose_vals = p['pose_keypoints_2d']   # get list of values (x1, y1 ,c1, x2, y2, c2 ...)
        s=people_ordering[o]
        people_ordering_new.append(s)
        i=0
        a=colorlist[s][i]
        b=colorlist[s][i+1]
        c=colorlist[s][i+2]
        for j in range(0, len(person_pose_vals), 3):
            x = int(person_pose_vals[j])
            y = int(person_pose_vals[j+1])
            c = person_pose_vals[j+2]
            
            img[(y-2):(y+2),(x-2):(x+2),:] = (a,b,c)  # do white
        
        #break  # remove to display for all the persons in the image
        
#    plt.figure()
#    plt.imshow(img)
    #return img
    return people_ordering_new

def mapping(pose_img1, pose_img2, arr):
        a=len(pose_img1['people'])
        b=len(pose_img2['people'])
        print(a)
        print(b)
        arr_map=[]
        
        for p in range(b):
            mini=10**1000
            
            for q in range(a):
                sums=0
                
                for r in range(0,75,3):
                    x2=pose_img2['people'][p]['pose_keypoints_2d'][r]
                    y2=pose_img2['people'][p]['pose_keypoints_2d'][r+1]
                    x1=pose_img1['people'][q]['pose_keypoints_2d'][r]
                    y1=pose_img1['people'][q]['pose_keypoints_2d'][r+1]
                    p1=np.array((x1,y1))
                    p2=np.array((x2,y2))
                    dist= np.linalg.norm(p2-p1)
                    sums= np.add(sums,dist)

                if(min(mini,sums)==sums):
                    mini=sums
                    s=q
            m=arr[s]
            arr_map.append(m)
            
        return arr_map


def plotPoseKeyPoints(img, pose_kp):
    '''
    img: input image RGB
    pose_kp: dictionary formed by openpose for the corresponding image
    '''
    people = pose_kp['people']
    
    for i, p in enumerate(people):
        person_pose_vals = p['pose_keypoints_2d']   # get list of values (x1, y1 ,c1, x2, y2, c2 ...)
        
        for j in range(0, len(person_pose_vals), 3):
            x = int(person_pose_vals[j])
            y = int(person_pose_vals[j+1])
            c = person_pose_vals[j+2]
            #print(x,y,c)
            img[(y-2):(y+2),(x-2):(x+2),:] = 255  # do white
        
        #break  # remove to display for all the persons in the image
        
    return img

def getPoseVector(pose_kp):
    """
    Receives the pose dictionary and returns the person keypoints as a vector
    """
    people = pose_kp['people']
    people_list = []
    for i, p in enumerate(people):
        person_pose_vals = p['pose_keypoints_2d']   # get list of values (x1, y1 ,c1, x2, y2, c2 ...)
        people_list.append(person_pose_vals)
#        for j in range(0, len(person_pose_vals), 3):
#            x = int(person_pose_vals[j])
#            y = int(person_pose_vals[j+1])
#            c = person_pose_vals[j+2]
#            #print(x,y,c)
#            img[(y-2):(y+2),(x-2):(x+2),:] = 255  # do white
        
        #break  # remove to display for all the persons in the image
        
    return np.array(people_list)      # return 2D matrix persons 
    


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
        if os.path.isfile(os.path.join(srcFolderPath, vid)) and vid.rsplit('.', 1)[1] in {'avi', 'mp4'}:
            infiles.append(os.path.join(srcFolderPath, vid))
            outfiles.append(os.path.join(destFolderPath, vid.rsplit('.', 1)[0]+".npy"))
            nFrames.append(getTotalFramesVid(os.path.join(srcFolderPath, vid)))
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
    return tot_frames


def getPoseFeats(DATASET, POSE_FEATS, videoName, onGPU):
    """
    Function to read all the frames of the video and get sequence of features
     one batch at a time. 
    This function can be called parallely called based on the amount of 
    memory available.
    
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
        poseFile = os.path.join(POSE_FEATS, prefix+'_{:012}'.format(frameCount)+'_keypoints.json')
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
        curr_frame = plotPoseKeyPoints(curr_frame, pose_dict)
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

