#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 02:30:05 2020

@author: arpan

@Description: Visualize predictions for Pose tracking of human actors.
"""

import _init_paths

import os
import sys
import cv2
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Blues8
from bokeh.embed import components
from utils import track_utils as utils

def writeCAMFramePoses(srcdataset, poses_df, destpath, write=True):
    """Write pose files by iterating on the dataframe rownames and saving paths 
    to a list to return. Used during bokeh based visualization.
    Parameters:
    ----------
    srcdataset : str
        path to the video dataset
    poses_df : pd.DataFrame
        dataframe containing the pose features (bounding boxes or pose vectors)
        with rownames as "videoname.avi_98_P0, videoname.avi_98_P1, ..."
    destpath : str
        path to folder where poses should be saved
    write : boolean
        Write the pose crops to the disk or not (True for writing and False for not)
        
    Returns:
    -------
        list of filenames written with complete path.
    """
    dest_filenames = []
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    bboxes = poses_df.shape[1] == 4
    # Iterate on all the poses in the dataset
    for rowname, values in poses_df.iterrows():
        vidname_pose = rowname.rsplit("_", 2)
        filename = vidname_pose[0]      # videoname with extension
        destfile = os.path.join(destpath, filename.rsplit(".",1)[0]+"_"+\
                                vidname_pose[1]+"_"+vidname_pose[2]+".png")
        # don't write the pose images, only append paths to list and return
        if write:
            img = utils.getImageFromName(srcdataset, rowname)
            # if bounding boxes are not available use the utils function
            if not bboxes:
                values = utils.getPoseBBoxes(values)
            img = img[int(values[1]):int(values[3]+1), int(values[0]):int(values[2]+1),...]        
            cv2.imwrite(destfile, img)
        dest_filenames.append(destfile)
        
    print("Poses written !")
    return dest_filenames


def write_stroke(srcDataPath, vidname, start, end, destpath, write=True):
    """ Write a Cricket stroke to a folder as mp4 format
    Parameters:
    -----------
    srcDataPath : str
        dataset path where src videos are kept
    vidname : str
        video name (with extension)
    start : int
        starting frame number of stroke
    end : int
        ending frame number of stroke
    destpath : str
        destination folder path where videos are to be written
    write : boolean
        whether to write the clip to disk or not.
    """
    srcvid = os.path.join(srcDataPath, vidname)
    assert os.path.isfile(srcvid), "Source video does not exist !"
    if not os.path.exists(destpath):
        os.makedirs(destpath)
    destfile = os.path.join(destpath, vidname.rsplit('.', 1)[0] \
                        +"_"+str(start)+"_"+str(end)+".mp4")
    
    if write:
        cap = cv2.VideoCapture(srcvid)
        if not cap.isOpened():
            print("Video Capture object not opened !")
            sys.exit(0)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(destfile, fourcc, fps, (160, 90), True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(end - start + 1):
            _, img = cap.read()
            img = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
            out.write(img)
        cap.release()
        out.release()
    
def visualize_prediction(srcDataPath, cluster_df, cluster_no=0, bboxes=True):
    '''
    Visualize a cluster's pose points on the dataset frames by visualizing bounding
    boxes on the corresponding video frames.
    Parameters:
    -----------
    srcDataPath : str
        dataset path where src videos are kept
    cluster_df : pd.DataFrame (P x 75)
        dataframe with pose vectors and rownames as videoname_frmNo_poseNo
    cluster_no : int
        The number appears on the visualization window. If the poses belong to 
        a cluster then that cluster no. can be passed.
    
    '''
    p_col = [255,127,127]  # a BGR color value for keypoints
    # Iterating on the rows of the dataframe
    for rowname, values in cluster_df.iterrows():
        # Read the image corresponding to rowname
        img = utils.getImageFromName(srcDataPath, rowname)
        if bboxes:
            x0, y0, x1, y1 = values
            cv2.rectangle(img, (x0, y0), (x1, y1), p_col, thickness=2)
                
        else:
            # Iterate over the keypoint coordinates 25 tuples of (x,y,c)
            for j in range(0, len(values), 3):
                x = int(values[j])
                y = int(values[j+1])
                #c = values[j+2]    # Not needed for visualization
                
                img[(y-2):(y+2),(x-2):(x+2),:] = p_col  # do Blue
        # Display on the window and wait for keystroke
        cv2.namedWindow("Cluster "+str(cluster_no))
        cv2.imshow("Cluster "+str(cluster_no), img)             
        utils.waitTillEscPressed()
    cv2.destroyAllWindows()

def plot_predictions_mask(srcDataPath, poses_df, cluster_ids, n_clusters, bboxes=True):
    '''
    Create a blank image of same size as original image and display poses as centroid 
    pixels with cluster pixels.
    '''
    img = utils.getImageFromName(srcDataPath, list(poses_df.index.values)[0])
    mask = np.zeros(img.shape, dtype='uint8')
    
    # [BLUE, GREEN, WHITE, RED, YELLOW, PINK, CYAN, BROWNISH-PINK, LIGHT GREEN, SKY BLUE]
    p_col = [[255,0,0], [0,255,0], [255,255,255], [0,0,255], [0,255,255], \
             [255,0,255], [255,255,0], [127,127,255], [127,255,127], [255,127,127]]
    for clust_no in range(n_clusters):
        cluster_df = poses_df.iloc[cluster_ids==clust_no]
        
        for rowname, values in cluster_df.iterrows():
            if bboxes:
                x_coord = [values[j] for j in range(0, len(values), 2) if values[j]!=0]
                y_coord = [values[j] for j in range(1, len(values), 2) if values[j]!=0]
            else:
                x_coord = [values[j] for j in range(0, len(values), 3) if values[j]!=0]
                y_coord = [values[j] for j in range(1, len(values), 3) if values[j]!=0]
                #conf = [values[j] for j in range(2, len(values), 3) if values[j]!=0]
            # Find center of a pose vector
            x_mean = np.mean(x_coord)
            y_mean = np.mean(y_coord)
    #        #cl_no = cluster_nos[idx]
            
            mask[int(y_mean), int(x_mean), :] = p_col[clust_no]
            #img[(y-2):(y+2),(x-2):(x+2),:] = p_col  # do Blue
#        people_ordering = utils.plotPersonBoundingBoxes(img, \
#                                self.vid_persons_bboxes[i], people_ordering, 1)
    cv2.namedWindow("Cluster "+str(n_clusters))
    cv2.imshow("Cluster "+str(n_clusters), mask)
    utils.waitTillEscPressed()
    cv2.destroyAllWindows()    



def plot_predictions_interactive(poses_df, cluster_ids, n_cluster, pose_paths, bboxes=True):
    
#    img = utils.getImageFromName(DATASET, list(poses_df.index.values)[0])
#    mask = np.zeros(img.shape, dtype='uint8')
    
#    source = ColumnDataSource(poses_df)
    # Format the tooltip
#    tooltips = [
#            ('Player', '@name'),
#            ('Three-Pointers Made', '@play3PM'),
#            ('Three-Pointers Attempted', '@play3PA'),
#            ('Three-Point Percentage', '@pct3PM{00.0%}')   
#           ]


    # Add Legend
#    p.legend.orientation = 'vertical'
#    p.legend.location = 'top_right'
#    p.legend.label_text_font_size = '10px'
    
#    hover = HoverTool()
#    
    # <div><img src="@Image" alt="" width="200" /></div>
#    hover.tooltips = """
#    <div>
#        <div><strong>Stroke: </strong>@rows</div>  
#        <div><img src="@Image" alt="" width="200" /></div>
#    </div>
#    """
#    p.add_tools(hover)
    
    # Get list from https://github.com/bokeh/bokeh/blob/master/bokeh/colors/named.py
    p_col = ["blue", "green", "red", "cyan", "brown", "lightblue", "lightgreen", "pink","yellow"]
    #colors = ["#%02x%02x%02x" % (int(r), int(g), 150) \
    #         for r, g in zip(50+2*x, 30+2*y) ]
    
    
    rows, x_mean, y_mean, colors = [], [], [], []
    
    for i, (rowname, values) in enumerate(poses_df.iterrows()):
        if bboxes:
            x_coord = [values[j] for j in range(0, len(values), 2) if values[j]!=0]
            y_coord = [values[j] for j in range(1, len(values), 2) if values[j]!=0]
        else:
            x_coord = [values[j] for j in range(0, len(values), 3) if values[j]!=0]
            y_coord = [values[j] for j in range(1, len(values), 3) if values[j]!=0]
            #conf = [values[j] for j in range(2, len(values), 3) if values[j]!=0]
        # Find center of a pose vector
        rows.append(rowname)
        x_mean.append(np.mean(x_coord))
        y_mean.append(np.mean(y_coord))
        colors.append(p_col[cluster_ids[i]])
            
    df = pd.DataFrame(data = {"stroke":rows, "x":x_mean, "y":y_mean, "color":colors, 
                              "pose": pose_paths})
    
    # <div><img src="@Image" alt="" width="200" /></div>
#         <video loop="true" autoplay="autoplay" controls>
#            <source src="v.mp4" type="video/mp4">
#            Your browser does not support the video tag.
#        </video>
    # <div><a href="0/-AfghanistanvsSouthAfrica-Match_1043_1251.avi" autostart="false" height="30" width="144">Video</a></div>
    source = ColumnDataSource(df)
    tooltips = """
    <div>
        <div><strong><b>Stroke: </b></strong>@stroke</div>  
        <div height="200">
        <div><img src="@pose" alt="" height="200" /></div>
        </div>
    </div>
    """
    #hover = HoverTool(tooltips = [("Stroke : ", "@stroke")])
    hover = HoverTool(tooltips = tooltips)
    
    output_file("poses.html")
    p = figure(
            plot_width=960,
            plot_height=540,
            title='Cluster Poses',
            x_axis_label='X Coordinate', 
            y_axis_label='Y Coordinate',
            x_range = [0, 640],
            y_range = [360, 0],
            tools="pan,box_select,zoom_in,zoom_out,save,reset"
        )
    
    # add a circle renderer with vectorized colors and sizes  legend_label=""+str(clust_no),
    p.circle(x="x", y="y", source=source, size=5, color="color", \
                  alpha=0.6)
    #tooltips = [("")]
    p.add_tools(hover)
    show(p)
