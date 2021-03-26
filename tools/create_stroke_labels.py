# -*- coding: utf-8 -*-
"""
Created on Wed Feb 3 13:54:51 2020

Description: Create Ground Truth and write to json file for SBD

@author: Bloodhound
"""

import cv2
import os
import json

def create_labels(srcFolderPath, strokes_path, destFolderPath, stop='all'):
    """
    Function to iterate over the videos in the subfolders and create stroke
    labels for them.
    Parameters:
    --------    
    srcFolderPath: str
        folder contain source videos (subfolders for main dataset)
    strokes_path: str
        path containing the json files for Cricket stroke locations
    destFolderPath: str
        folder where the labels are to be saved
    
    """
    assert os.path.isdir(srcFolderPath), "Source Directory not found."
    
    # iterate over the videos in srcFolderPath and label for each video 
    vfiles = sorted(os.listdir(srcFolderPath))
    # create destination path to store the files
    if not os.path.exists(destFolderPath):
        os.makedirs(destFolderPath)
    
    traversed_tot = 0
    # iterate over the video files inside the directory 
    for f in vfiles:
        srcVideo = os.path.join(srcFolderPath, f)
        stroke_file = os.path.join(strokes_path, f.rsplit('.', 1)[0] + '.json')
        assert os.path.isfile(stroke_file), "Stroke file missing."
        with open(os.path.join(strokes_path, stroke_file), 'r') as fp:
            strokes = json.load(fp)
            
        # get list of strokes
        strokes = strokes[list(strokes.keys())[0]]
        
        labels = getStrokeLabelsForVideo(srcVideo, strokes)
        
        # save at the destination, if extracted successfully
        if not labels is None:
            destFile = os.path.join(destFolderPath, f.rsplit('.',1)[0])+".json"
            with open(destFile, "w") as fp:
                json.dump({f : labels}, fp)
            traversed_tot += 1
            print("Done "+str(traversed_tot)+" : "+f)
        else:
            print("Labels file not created !!")
        
        # to stop after successful traversal of 2 videos, if stop != 'all'
        if stop != 'all' and traversed_tot == stop:
            break
        
    print("No. of files written to destination : "+str(traversed_tot))
    if traversed_tot == 0:
        print("Check the structure of the dataset folders !!")
    
    return traversed_tot


# create cricket shot labels for single video, params are srcVideo. Returns list of
# tuples like (starting_frame, ending_frame)
def getStrokeLabelsForVideo(srcVideo, strokes):
    '''
    Parameters:
    -----------
    srcVideo : str
        complete path to the source video 
    stroke : list of tuples
        list of tuples with starting and ending stroke frame locations
        
    Return:
    
    '''
    
    cap = cv2.VideoCapture(srcVideo)
    shotLabels = []
    i = 0
    isShot = False
    print("Video : {}".format(srcVideo))
    if not cap.isOpened():
        print("Could not open the video file !! Abort !!")
        return None
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Iterate on the stroke labels
    for start, end in strokes:
        # Iterate over the frames of a single stroke
        for i in range(start, end+1):
            print("Stroke : {} :: Frame : {}".format((start, end), i))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Current", frame)
                assigned_label = waitTillEscPressed()
            else:
                break
                
    print("Stroke Labels : {}".format(shotLabels))
    cap.release()
    cv2.destroyAllWindows()
    return shotLabels
    
#    while cap.isOpened() and i<=(length+2):
#        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#        ret, frame = cap.read()
#        if ret:
#            cv2.imshow("Prev",frame)
#            ret, frame = cap.read()
#            if ret:
#                print("Prev : "+str(i)+" ## Next : "+str(i+1)+" ## Shot : " \
#                                   +str(isShot)+"  / "+str(length))
#                cv2.imshow("Next", frame)
#            else:
#                print("Next frame is NULL")
#            direction = waitTillEscPressed()
#            if direction == 1:
#                i +=1
#                shotLabels.append(isShot)
#            elif direction == 0:
#                i -=1
#                shotLabels.pop()
#            elif direction == 2:
#                if not isShot:
#                    isShot = True
#                else:
#                    print("Shot already started. Press 'b' to move back and edit.")
#                #shotLabels.append(isShot)
#            elif direction == 3:
#                if isShot:
#                    isShot = False
#                else:
#                    print("Shot not started yet. Press 'b' to move back and edit.")
#                
#        else:
#            break
#    
#    shots_lst = getListOfShots(shotLabels)
#
#    print("No. of cricket shots in video : "+str(len(shots_lst)))
#    print(shots_lst)
#    print("Total no of frames traversed : ")
#    print(i)
#    cap.release()
#    cv2.destroyAllWindows()
#    
#    return shots_lst

# get the starting and ending indices from the list of values.
def getListOfShots(shotLabels):
    shots = []
    start, end = -1, -1
    for i,isShot in enumerate(shotLabels):
        if isShot:
            if start<0:     # First True after a sequence of False
                start = i+1   
        else:
            if start>0:     # First false after a sequence of True
                end = i
                shots.append((start,end))
                start,end = -1,-1
    return shots

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
        # start of shot
        elif cv2.waitKey(0)==115:
            print("'s' pressed. Start of shot.")
            return 2
        # end of shot
        elif cv2.waitKey(0)==102:
            print("'f' pressed. End of shot.")
            return 3


if __name__=='__main__':
    
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    destFolder = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/data/stroke_types"
    
#    with open(meta_filepath, 'r') as fp:
#        meta_info = json.load(fp)
#        
#    train_meta_info = [k for k in meta_info.items if k[1]['partition']=='training']
#    tr1 = [t[0] for t in tr]
#    tr2 = [t[1]['nFrames'] for t in tr]
#    import pandas as pd
#    p = pd.DataFrame({'vNames':tr1, 'nFrames':tr2})
#    p = p.sort_values(by='nFrames')
#    p = p.reset_index(drop=True)

    create_labels(DATASET, LABELS, destFolder, 4)
    
    