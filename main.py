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
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Set the path of the input videos and the openpose extracted features directory
# Server Paths
DATASET = "/opt/datasets/cricket/ICC_WT20"
POSE_FEATS = "/home/arpan/cricket/output_json"
LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"

# Local Paths
if not os.path.exists(DATASET):
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    POSE_FEATS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/output_json"
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"



class VideoPoseTrack:
    
    def __init__(self, datasetPath, srcVid, posesPath, strokesFilePath):
        """
        Initializing the path variables and assertions for paths.
        """
        self.datasetPath = datasetPath
        self.srcVid = srcVid 
        self.posesPath = posesPath
        self.strokesFilePath = strokesFilePath
        srcVideoPath = os.path.join(datasetPath, srcVid)
        assert os.path.exists(srcVideoPath), "{} does not exist!".format(srcVideoPath)
        assert os.path.exists(posesPath), "{} does not exist!".format(posesPath)
        assert os.path.exists(strokesFilePath), "{} does not exist!".format(strokesFilePath)
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
        self.vid_persons = []
        for i in range(self.nFrames):
            poseFile = self.srcVid.rsplit(".", 1)[0]+"_{:012}".format(i)+"_keypoints.json"
            poseFile = os.path.join(self.posesPath, poseFile)
            with open(poseFile, 'r') as fp:
                frame_persons_list = json.load(fp)
            # form a list of 2D matrices.
            self.vid_persons.append(utils.getPoseVector(frame_persons_list))
        
        print("Read keypoint vectors")
        return self.vid_persons
    
    def getPoseBBoxes(self):
        """
        return the list of pose bounding box vectors for the video. 
        For N frames, get N matrices of shape (n_persons_in_frame, 4), 
        where 4 is the sequence of (x0, y0, x1, y1) coordinates.
        """
        self.vid_persons_bboxes = []
        for i in range(self.nFrames):
            poseFile = self.srcVid.rsplit(".", 1)[0]+"_{:012}".format(i)+"_keypoints.json"
            poseFile = os.path.join(self.posesPath, poseFile)
            with open(poseFile, 'r') as fp:
                frame_persons_list = json.load(fp)
            # form a list of 2D matrices.
            self.vid_persons_bboxes.append(utils.getPoseBBoxes(frame_persons_list))
        
        print("Read keypoint vectors")
        return self.vid_persons_bboxes
        
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
                    
        
    def visualizeVideoWithPoses(self):
        """
        Visualize by plotting the points in the frames.
        """
        global refpt, clicked
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
                while(True):
                
                    cv2.setMouseCallback("Poses", get_click_coordinates)
                    
                    # get the box corresponding to the click coordinates
                    if clicked:
                        box = utils.highlight_selected_box(img, self.vid_persons_bboxes[i],\
                                                 people_ordering, refpt)
                        print("Box : {}".format(box))
                        clicked = False
                        cv2.imshow("Poses", img)
                    
                    # For moving forward
                    if cv2.waitKey(0)==27:
                        print("Esc Pressed. Move Forward.")
                        i+=1
                        break
                
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

refpt, clicked = None, False
def get_click_coordinates(event, x, y, flags, param):
    # grab references to the global variables
    global refpt, clicked
    # if the left mouse button was pressed or released, record (x, y) coordinates 
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
        refpt = (x, y)
        print("coordinates : {}".format((x,y)))
        clicked = True
        # draw a rectangle around the region of interest
        #cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        #cv2.imshow("image", image)

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
    sizes = [utils.getTotalFramesVid(os.path.join(DATASET, f+".avi")) for f in train_lst]
    print("Size : {}".format(sizes))
    #hlDataset = VideoDataset(tr_labs, sizes, seq_size=SEQ_SIZE, is_train_set = True)
    #print hlDataset.__len__()
    
    # Create object for one video only
    v = VideoPoseTrack(DATASET, train_lst[0]+".avi", POSE_FEATS, tr_labs[0])
    vid_poses = v.getPoseBBoxes()
    v.visualizeVideoWithPoses()
    
    

    