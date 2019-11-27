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
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.vq import vq


# Set the path of the input videos and the openpose extracted features directory
# Server Paths
DATASET = "/opt/datasets/cricket/ICC_WT20"
POSE_FEATS = "/home/arpan/cricket/output_json"
BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"

# Local Paths
if not os.path.exists(DATASET):
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    POSE_FEATS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/output_json"
    BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"



class VideoPoseTrack:
    
    def __init__(self, datasetPath, srcVid, posesPath, strokesFilePath, gt_path, gtFile):
        """
        Initializing the path variables and assertions for paths.
        Params:
        ------
        datasetPath: path containing the videos
        srcVid: name of the video
        posesPath: path containing the json pose files
        strokesFilePath: filepath for JSON file with stroke location labels
        gt_path: path with csv files of batsman labels
        gtFile: filename for batsman labels for current video
            
        """
        self.datasetPath = datasetPath
        self.srcVid = srcVid 
        self.posesPath = posesPath
        self.strokesFilePath = strokesFilePath
        self.gt_path = gt_path
        self.gtFile = gtFile
        srcVideoPath = os.path.join(datasetPath, srcVid)
        assert os.path.exists(srcVideoPath), "{} does not exist!".format(srcVideoPath)
        assert os.path.exists(posesPath), "{} does not exist!".format(posesPath)
        assert os.path.exists(strokesFilePath), "{} does not exist!".format(strokesFilePath)
        assert os.path.exists(os.path.join(gt_path, gtFile)), "{} does not exist".format(gtFile)
        self.nFrames = int(utils.getTotalFramesVid(srcVideoPath))
        assert self.checkPoseFiles(), "Poses JSON Files missing for {}".format(srcVid)
        # read the list of stroke labels from the JSON file. Dict with one key
        with open(strokesFilePath, 'r') as fp:
            strokes_dict = json.load(fp)
        # get list of strokes (list of tuples)
        self.strokes = strokes_dict[list(strokes_dict.keys())[0]]

        
    def checkPoseFiles(self):
        """
        Check whether all the JSON files for pose keypoints is present for srcVid
        or not. Used in assertion.
        """
        for i in range(self.nFrames):
            poseFile = self.srcVid.rsplit(".", 1)[0]+"_{:012}".format(i)+"_keypoints.json"
            if not os.path.exists(os.path.join(self.posesPath, poseFile)):
                print("{} missing".format(poseFile))
                return False
        return True
    
    def getPoseFeatures(self):
        """
        return the list of pose feature vectors for the video. For N frames, get
        N matrices of shape (n_persons_in_frame, 75), where 75 is the sequence of
        25 (x, y, confidence) pose keypoints.
        """
        self.poseVecs = []
        for i in range(self.nFrames):
            poseFile = self.srcVid.rsplit(".", 1)[0]+"_{:012}".format(i)+"_keypoints.json"
            poseFile = os.path.join(self.posesPath, poseFile)
            with open(poseFile, 'r') as fp:
                frame_persons_list = json.load(fp)
            # form a list of 2D matrices.
            self.poseVecs.append(utils.getPoseVector(frame_persons_list))
        
        print("Read keypoint vectors")
        return self.poseVecs
    
    def getPoseBBoxes(self):
        """
        return the list of pose bounding box vectors for the video. 
        For N frames, get N matrices of shape (n_persons_in_frame, 4), 
        where 4 is the sequence of (x0, y0, x1, y1) coordinates.
        """
        self.poseBBoxes = []
        for i in range(self.nFrames):
            poseFile = self.srcVid.rsplit(".", 1)[0]+"_{:012}".format(i)+"_keypoints.json"
            poseFile = os.path.join(self.posesPath, poseFile)
            with open(poseFile, 'r') as fp:
                frame_persons_list = json.load(fp)
            # form a list of 2D matrices.
            self.poseBBoxes.append(utils.getPoseBBoxes(frame_persons_list))
        
        print("Read keypoint vectors")
        return self.poseBBoxes
        
    def getGTBatsmanPoses(self):
        """
        return the list of pose bounding box vectors for the batsman. 
        For frames in trimmed videos, get vectors shape (4,), 
        where 4 is the sequence of (x0, y0, x1, y1) coordinates of batsman.
        """
        self.gt_batsman = {}
        import csv
        with open(os.path.join(self.gt_path, self.gtFile), 'r') as fp:
            data = csv.reader(fp)
            for row in data:
                if type(row)==list and len(row)>0 and row[-1]=='Batsman':
                    # save in {FrameNo : [x0, y0, x1, y1]} format
                    self.gt_batsman[int(row[0])] = [int(row[2]), int(row[3]), \
                                    int(row[4]), int(row[5])]
        return self.gt_batsman
    
    def getCAMFramePoses(self, first_n = 1):
        """
        Read the poses of first n frames in CAM1, stack them and return 
        strokesPath: path to the json files with temporal stroke locations
        """
        assert hasattr(self, 'poseBBoxes'), "Execute getPoseBBoxes()"
        
        if not hasattr(self, 'poseVecs'):
            poses = self.getPoseFeatures()
        
        topn_poses = []
        # Iterate over top n frames of CAM1 and select those vecs
        for (start, end) in self.strokes:
            if (start+first_n) <= end:
                topn_poses.append(poses[start:(start+first_n)])
            
        return topn_poses
            
        
        
    
    def disambiguatePersons(self):
        """
        Identify the persons by disambiguation. Apply Bipartite Graph Matching.
        Take list of matrices and for each consecutive pair of them, get the 
        trajectory of people. Tracks of people over the frames.
        Where size is (0,) for the list item, ignore (break: person not present)
        
        """
        track_labels = []
        start = True
        # Iterate over the frame pose matrices
        for i, poses in enumerate(self.vid_persons):
            
            if poses.shape == (0,):     # no person in frame
                track_labels.append([])
                start = True
                continue
#            else:
#                if start:
#                    #assign new labels
                    
    def visualizeVideoWithBatsman(self):
        """
        Visualize by plotting the points in the frames.
        """
        
        for start, end in self.strokes:
            i = start
            while i<=end:
                poseFile = self.srcVid.rsplit(".", 1)[0]+"_{:012}".format(i)+"_keypoints.json"
                # read the image from the datasetPath only using poseFile name
                img=utils.getImageFromName(self.datasetPath, poseFile)
                assert hasattr(self, 'gt_batsman'), "Execute getGTBatsmanPoses()"
                # draw the pose if the ground truth is available
                if i in self.gt_batsman.keys():
                    [x0, y0, x1, y1] = self.gt_batsman[i]
                    print("Frame : {} :: Boxes : {}".format(i, [x0, y0, x1, y1]))
                    cv2.rectangle(img, (x0, y0), (x1, y1), [0,255,0], thickness=1)
                
                cv2.imshow("Poses", img)
                
                # For moving forward
                if cv2.waitKey(0)==27:
                    print("Esc Pressed. Move Forward.")
                i+=1
                #break
        cv2.destroyAllWindows()


        
    def visualizeVideoWithPoses(self):
        """
        Visualize by plotting the points in the frames.
        """
        global refpt, img, vbboxes, pOrder
        #Iterate over the strokes
        for start, end in self.strokes:
            i = start
            while i<=end:
                poseFile = self.srcVid.rsplit(".", 1)[0]+"_{:012}".format(i)+"_keypoints.json"
                # read the image from the datasetPath only using poseFile name
                img=utils.getImageFromName(self.datasetPath, poseFile)
                assert hasattr(self, 'vid_persons_bboxes'), "Execute getPoseFeatures()"
#                # read the poses from JSON
#                with open(os.path.join(self.posesPath, poseFile), 'r') as fp:
#                    pose_img = json.load(fp)
                people_ordering = list(range(len(self.vid_persons_bboxes)))
#                people_ordering = utils.plotPersonFromMatrix(img, \
#                                        self.vid_persons[i], people_ordering)
                people_ordering = utils.plotPersonBoundingBoxes(img, \
                                        self.vid_persons_bboxes[i], people_ordering, 1)
                
                cv2.namedWindow("Poses")
                cv2.imshow("Poses", img)
                
                vbboxes = self.vid_persons_bboxes[i]
                pOrder = people_ordering
                
                
                cv2.setMouseCallback("Poses", get_click_coordinates)
                    
                # get the box corresponding to the click coordinates
                #if clicked:
                    #box = utils.highlight_selected_box(img, self.vid_persons_bboxes[i],\
                    #                             people_ordering, refpt)
                    #print("Box : {}".format(box))
                    #clicked = False
                    #cv2.imshow("Poses", img)
                    
                # For moving forward
                if cv2.waitKey(0)==27:
                    print("Esc Pressed. Move Forward.")
                    i+=1
                    #break
                
#                direction = waitTillEscPressed()
#                if direction == 1:
#                    i +=1
#                elif direction == 0:
#                    i -=1
#                elif direction == 2:
#                    break
#                elif direction == 3:
#                    break

#            
#            for l in range(j,k):           #tracking starts
#                print(l)
#                print(l+1)
#                with open(os.path.join(POSE_FEATS, all_poses[l]), 'r') as fp:
#                    pose_img1 = json.load(fp)
#                with open(os.path.join(POSE_FEATS, all_poses[l+1]), 'r') as fp:
#                    pose_img2 = json.load(fp)
#                
#                arr21 = utils.mapping(pose_img1, pose_img2, arr)
#                
#                print(arr21)
#                img=utils.getImageFromName(all_poses[l+1])
#                arr=utils.plotPerson(img, pose_img2,arr21)
            break
        cv2.destroyAllWindows()
        
    def getPoseMotionFeatures(self):
        """
        Retrieve the L2 difference features of consecutive poses.
        Returns (N-1) list of matrices using the poses matrix.
        """
        
        pass

img, refpt, pOrder, vbboxes = None, None, None, None
def get_click_coordinates(event, x, y, flags, param):
    # grab references to the global variables
    global refpt, img, pOrder, vbboxes
    # if the left mouse button was pressed or released, record (x, y) coordinates 
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
        refpt = (x, y)
        print("coordinates : {}".format((x,y)))
        #clicked = True
        v = img.copy()
        box = utils.highlight_selected_box(v, vbboxes, pOrder, refpt)
        
        # draw a rectangle around the region of interest
        #cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("Poses", v)


def waitTillEscPressed():
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


def get_stacked_poses_df(train_lst, tr_labs, tr_gt_batsman, n=1):
    '''
    Create a dataframe of the first n frame poses with vidname_frameNo as rownames.
    '''
    poses_lst = []
    # Create object for one video only
    for i, vid_file in enumerate(train_lst):
        
        v = VideoPoseTrack(DATASET, train_lst[i]+".avi", POSE_FEATS, tr_labs[i], BAT_LABELS, tr_gt_batsman[i])
        vid_poses = v.getPoseBBoxes()
        #v.visualizeVideoWithPoses()
        #batsman_poses = v.getGTBatsmanPoses()
        #v.visualizeVideoWithBatsman()
        topn_poses = v.getCAMFramePoses(first_n = n)
        
        for idx, (beg, end) in enumerate(v.strokes):
            for frm_no in range(n):
                pose_arr = topn_poses[idx][frm_no]
                rownames = []
                for pose_no in range(pose_arr.shape[0]):
                    rownames.append(train_lst[i]+"_"+str(beg+frm_no)+"_P"+str(pose_no))
                poses_lst.append(pd.DataFrame(pose_arr, index=rownames))
        #break
    return pd.concat(poses_lst)

def kmeans(flows, clusters=4):
    km = KMeans(n_clusters=clusters, algorithm='elkan', random_state=0)
    km.fit(flows)
    return km

def dbscan(flows, clusters=4, epsilon=100):
    db = DBSCAN(eps=epsilon, min_samples=10).fit(flows)
    return db

def visualize_prediction(cluster_df, cluster_no=0):
    
    for rowname, values in cluster_df.iterrows():
        
        img = utils.getImageFromName(DATASET, rowname)
        #cl_no = cluster_nos[idx]
        p_col = [255,255,255]
        for j in range(0, len(values), 3):
            x = int(values[j])
            y = int(values[j+1])
            c = values[j+2]
            
            img[(y-2):(y+2),(x-2):(x+2),:] = p_col  # do Blue
#        people_ordering = utils.plotPersonBoundingBoxes(img, \
#                                self.vid_persons_bboxes[i], people_ordering, 1)
        cv2.namedWindow("Cluster "+str(cluster_no))
        cv2.imshow("Cluster "+str(cluster_no), img)
                
        waitTillEscPressed()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    
    all_poses = os.listdir(POSE_FEATS)[:5]    # taking only 5 json files

    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
    train_lab = [f+".json" for f in train_lst]
    val_lab = [f+".json" for f in val_lst]
    test_lab = [f+".json" for f in test_lst]
    train_gt_batsman = [f+"_gt.txt" for f in train_lst]
    val_gt_batsman = [f+"_gt.txt" for f in val_lst]
    test_gt_batsman = [f+"_gt.txt" for f in test_lst]
    
    
    #####################################################################
    
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    tr_gt_batsman = [os.path.join(BAT_LABELS, f) for f in train_gt_batsman]
    sizes = [utils.getTotalFramesVid(os.path.join(DATASET, f+".avi")) for f in train_lst]
    print("Size : {}".format(sizes))
    #hlDataset = VideoDataset(tr_labs, sizes, seq_size=SEQ_SIZE, is_train_set = True)
    #print hlDataset.__len__()
    poses_df = get_stacked_poses_df(train_lst, tr_labs, train_gt_batsman, n=1)
            
    km_poses = kmeans(poses_df, clusters=5)
    
    clust_centers = km_poses.cluster_centers_
    word_ids = vq(poses_df, clust_centers)[0]  # ignoring the distances in [1]
    
    clust_no = 2
    visualize_prediction(poses_df.iloc[word_ids==clust_no], cluster_no=clust_no)

    # clust_no = 2 is for persons near and corresponding to batsman
