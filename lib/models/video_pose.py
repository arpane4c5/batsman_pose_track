#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 05:40:09 2020

@author: arpan

@Description: Class defined for reading the Openpose extracted poses for a video and 
related functions for visualization and tracking.
"""
import _init_paths

import os
import json
import cv2
import numpy as np
from utils import track_utils as utils
from pylab import show, ogrid, gca
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png


class VideoPoseTrack:
    
    def __init__(self, datasetPath, srcVid, posesPath, strokesFilePath):
        """
        Initializing the path variables and assertions for paths.
        Params:
        ------
        datasetPath: str
            path containing the videos
        srcVid: str
            name of the video with extension
        posesPath: str
            path containing the json pose files
        strokesFilePath: str
            complete path with filename for JSON file containing stroke location labels
            
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
        Check whether all the JSON files for pose keypoints are present for srcVid
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
        Return the list of pose feature vectors for the video. For N frames, get
        N matrices of shape (n_persons_in_frame, 75), where 75 is the sequence of
        25 (x, y, confidence) pose keypoints.
        
        """
        if hasattr(self, "poseVecs"):
            return self.poseVecs
        self.poseVecs = []
        for i in range(self.nFrames):
            poseFile = self.srcVid.rsplit(".", 1)[0]+"_{:012}".format(i)+"_keypoints.json"
            poseFile = os.path.join(self.posesPath, poseFile)
            with open(poseFile, 'r') as fp:
                frame_persons_list = json.load(fp)
            # form a list of 2D matrices.
            self.poseVecs.append(utils.getPoseVector(frame_persons_list))
        
        print("Read : {}".format(self.srcVid))
        return self.poseVecs
    
    def getPoseBBoxes(self):
        """
        Return the list of pose bounding box vectors for the video. 
        For N frames, get N matrices of shape (n_persons_in_frame, 4), 
        where 4 is the sequence of (x0, y0, x1, y1) coordinates.
        """
        if hasattr(self, 'poseBBoxes'):
            return self.poseBBoxes
        if not hasattr(self, 'poseVecs'):
            self.getPoseFeatures()
            
        self.poseBBoxes = [utils.convertPoseVecsToBBoxes(frm_poses) for frm_poses \
                               in self.poseVecs]
        print("Created Bounding Boxes")
        return self.poseBBoxes
        
    
    def getCAMFramePoses(self, first_n = 1, bounding_box=False):
        """
        Read the poses of first n frames in CAM1, stack them and return 
        strokesPath: path to the json files with temporal stroke locations
        """
        if not hasattr(self, 'poseBBoxes'):
            self.getPoseBBoxes()
        
        topn_poses = []
        # Iterate over top n frames of CAM1 and select those vecs
        for (start, end) in self.strokes:
            if (start+first_n) <= end:
                if bounding_box:
                    topn_poses.append(self.poseBBoxes[start:(start+first_n)])
                else:
                    topn_poses.append(self.poseVecs[start:(start+first_n)])
            
        return topn_poses
                    
    def visualizeVideoWithBatsman(self):
        """
        Visualize by plotting the points in the frames.
        """
        for start, end in self.strokes:
            i = start
            while i<=end:
                poseFile = self.srcVid+"_{:012}".format(i)+"_keypoints.json"
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
        
    def visualizeVideoWithPoses(self, bboxes=True):
        """
        Visualize by plotting the points in the frames.
        """
        global refpt, img, pose_mat, pOrder
        
        if not hasattr(self, 'poseBBoxes'):
            self.getPoseBBoxes()
        
        #Iterate over the strokes
        for start, end in self.strokes:
            i = start
            while i<=end:
                poseFile = self.srcVid+"_{:012}".format(i)+"_keypoints.json"
                people_ordering = list(range(self.poseBBoxes[i].shape[0])) if bboxes \
                                    else list(range(self.poseVecs[i].shape[0]))
                pose_mat = self.poseBBoxes[i] if bboxes else self.poseVecs[i]
                
                # read the image from the datasetPath only using poseFile name
                img=utils.getImageFromName(self.datasetPath, poseFile)
                
                if bboxes:
                    utils.plotPersonBoundingBoxes(img, self.poseBBoxes[i], \
                                              people_ordering, 1)
                else:
                    utils.plotPersonFromMatrix(img, self.poseVecs[i], people_ordering)
                
                cv2.namedWindow("Poses")
                cv2.imshow("Poses", img)
                
                pOrder = people_ordering
                cv2.setMouseCallback("Poses", get_click_coordinates)
                    
                # For moving forward
                if cv2.waitKey(0)==27:
                    i+=1
                    #break
#                direction = utils.waitTillEscPressed()
#                if direction == 1:
#                    i +=1
#                elif direction == 0:
#                    i -=1
#                elif direction == 2:
#                    break
#                elif direction == 3:
#                    break
            break
        cv2.destroyAllWindows()
        
    def track_poses(self, epsilon=50, bboxes=True, visualize=True):
        '''Iterate over the strokes and track bounding boxes
        '''
        if not hasattr(self, 'poseBBoxes'):
            self.getPoseBBoxes()
        track_count = 0
        vid_tracks = []
                
        #Iterate over the strokes
        for start, end in self.strokes:
            i = start
            stroke_tracks = []
            prev_poses = self.poseBBoxes[i-1] if bboxes else self.poseVecs[i-1]
            prev_ordering = list(range(track_count, track_count + prev_poses.shape[0]))
            while i<=end:
                curr_poses = self.poseBBoxes[i] if bboxes else self.poseVecs[i]
                
                curr_ordering = utils.get_next_mapping(prev_poses, curr_poses, \
                                                    prev_ordering, track_count, epsilon)
                if visualize:
                    poseFile = self.srcVid+"_{:012}".format(i)+"_keypoints.json"
                    # read the image from the datasetPath only using poseFile name
                    img=utils.getImageFromName(self.datasetPath, poseFile)
                
                    print(curr_ordering)
                    if bboxes:
                        utils.plotPersonBoundingBoxes(img, self.poseBBoxes[i], curr_ordering, 1)
                    else:
                        utils.plotPersonFromMatrix(img, self.poseVecs[i], curr_ordering)
                
                    cv2.namedWindow("Poses")
                    cv2.imshow("Poses", img)
                    # For moving forward
                    if cv2.waitKey(0)==27:
                        print("Esc Pressed. Move Forward.")
                
                if curr_ordering != []:
                    if max(curr_ordering) >= track_count:
                        track_count = max(curr_ordering) + 1
                prev_poses = curr_poses
                prev_ordering = curr_ordering
                stroke_tracks.append(curr_ordering)
                i+=1
            vid_tracks.append(stroke_tracks)
            # break  # uncomment to run for all the strokes in a video
        print("Total No of tracks : {}".format(track_count))
        cv2.destroyAllWindows()
        self.vid_tracks = vid_tracks
        self.track_count = track_count
        return vid_tracks, track_count
            
    def find_longest_track(self, epsilon=50, bboxes=True, visualize=True):
        '''
        '''
        if not hasattr(self, "vid_tracks"):
            self.track_poses(epsilon, bboxes, visualize=False)
        
        tracks = []
        for idx, (start, end) in enumerate(self.strokes):
            stroke_tracks = self.vid_tracks[idx]
            tracks.append(utils.get_track2frame(stroke_tracks))
        
        self.tracks = tracks
        tr_len, pose_ids, stroke_st = [], [], []
        # find longest track by iterating over the strokes
        for idx, track in enumerate(tracks):
            length, pid, stroke_idx  = -1, -1, -1
            # Iterate over the dictionary values of a stroke
            for k, v in track.items():
                if len(v) > length:
                    length = len(v)
                    pid = k
                    stroke_idx = idx
            tr_len.append(length)
            pose_ids.append(pid)
            stroke_st.append(self.strokes[stroke_idx])
            
        return tr_len, pose_ids, stroke_st
            
    
    def get_normalized_poses(self, height=360, width=640):
        '''
        Represent the poses as keypoints in range [0, 1]
        '''
        poseVecsNorm = []
        for frm_poses in self.poseVecs:
            poseVecsNorm.append(utils.normalize_pose(frm_poses))
            
        self.poseVecsNorm = poseVecsNorm
        return poseVecsNorm
    
    def label_tracks(self, epsilon=50, bboxes=True, visualize=True):
        '''Visualize complete track and label
        '''
        if not hasattr(self, "vid_tracks"):
            self.track_poses(epsilon, bboxes, visualize=False)
            
    def plot_tracks(self, bat_labels, tr_gt_batsman, bboxes=True):
        if not hasattr(self, "gt_batsman"):
            self.gt_batsman = utils.getGTBatsmanPoses(bat_labels, tr_gt_batsman)
        #vid_file : utils.getGTBatsmanPoses(BAT_LABELS, tr_gt_batsman[i])})
        ax = gca(projection='3d')
        x, z = ogrid[0:int(360/5), 0:int(640/5)]
        y = np.atleast_2d(10.0)
        for start, end in self.strokes:
            i = start
            while i<=end:
                poseFile = self.srcVid+"_{:012}".format(i)+"_keypoints.json"
                # read the image from the datasetPath only using poseFile name
                img=utils.getImageFromName(self.datasetPath, poseFile)
                w, h = img.shape[1], img.shape[0]
                # draw the pose if the ground truth is available
                if i in self.gt_batsman.keys():
                    [x0, y0, x1, y1] = self.gt_batsman[i]
                    print("Frame : {} :: Boxes : {}".format(i, [x0, y0, x1, y1]))
                    cv2.rectangle(img, (x0, y0), (x1, y1), [0,255,0], thickness=2)
                    
                img = cv2.resize(img, (int(w/5), int(h/5)), interpolation = cv2.INTER_AREA) 
                norm_img = cv2.normalize(img, None, alpha=0, beta=1, \
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
                ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=norm_img)
                
                i+=1
                break
#        fn = get_sample_data("lena.png", asfileobj=False)
#        img = read_png(fn)
        
            break
        show()
        
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
        
    def getPoseMotionFeatures(self):
        """
        Retrieve the L2 difference features of consecutive poses.
        Returns (N-1) list of matrices using the poses matrix.
        """
        pass
    
    
img, refpt, pOrder, vbboxes = None, None, None, None
def get_click_coordinates(event, x, y, flags, param):
    # grab references to the global variables
    global refpt, img, pOrder, pose_mat
    # if the left mouse button was pressed or released, record (x, y) coordinates 
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONUP:
        refpt = (x, y)
        print("coordinates : {}".format((x,y)))
        #clicked = True
        v = img.copy()
        utils.highlight_selected_box(v, pose_mat, pOrder, refpt)
        
        # draw a rectangle around the region of interest
        #cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("Poses", v)
