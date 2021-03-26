#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:09:32 2018

@author: Arpan
Refer: https://github.com/hunkim/PyTorchZeroToAll/blob/master/name_dataset.py
"""

# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import json
import os
from skimage import transform, io
from skimage.transform import resize
from torchvision.transforms import functional as F


class VideoDataset(Dataset):
    """ Cricket Strokes dataset."""

    # Initialize your data, download, etc.
    # vidsList: List of paths to labels files containing action instances
    def __init__(self, featuresPath, stroke_files, vidsSizes, seq_size=10, stride=1, \
                 transform=None, is_train_set=False):
        
        assert os.path.isdir(featuresPath), "{} does not exist.".format(featuresPath)
        # Check seq_size is valid ie., less than min action size
        self.featuresPath = featuresPath
        self.keys = []
        self.frm_sequences = []
        self.transform = transform
        #self.labels = []
        
        # read files and get training, testing and validation partitions
#        self.shots = [] # will contain list of dictionaries with vidKey:LabelTuples
        for i, labfile in enumerate(stroke_files):
            assert os.path.exists(labfile), "Path does not exist"
            with open(labfile, 'r') as fobj:
                shots = json.load(fobj)
                
            k = list(shots.keys())[0]
            strokes = shots[k]
            for (start, end) in strokes:
                
                for t in range(start, end+1, stride):
                    # add a clip tuple to the list of windows
                    # (start, end) frame no
                    self.frm_sequences.append((t, t+seq_size-1))
                    # file names (without full path), only keys
                    self.keys.append(k)
            
        self.len = len(self.keys)
        self.seq_size = seq_size

    def __getitem__(self, index):
        '''
        '''
        vidFile = os.path.join(self.featuresPath, (self.keys[index].rsplit('/', 1))[1])
        vidFile = vidFile.rsplit('.', 1)[0] + '.npy'
        start, end = self.frm_sequences[index]
        frame_inputs = np.load(vidFile, mmap_mode='r')[start:end]
        return frame_inputs, self.keys[index], self.frm_sequences[index]

    def __len__(self):
        return self.len

    def __seq_size__(self):
        return self.seq_size


def get_sport_clip(frames_list, verbose=True):
    """
    Loads a clip to be fed to C3D for classification.
    TODO: should I remove mean here?
    
    Parameters
    ----------
    clip_name: str      OR frames_list : list of consecutive N framePaths
        the name of the clip (subfolder in 'data').
    verbose: bool
        if True, shows the unrolled clip (default is True).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fr, h, w).
    """

    #clip = sorted(glob(os.path.join('data', clip_name, '*.jpg')))
    clip = np.array([resize(io.imread(frame), output_shape=(112, 200), preserve_range=True) for frame in frames_list])
    clip = clip[:, :, 44:44+112, :]  # crop centrally

    if verbose:
        clip_img = np.reshape(clip.transpose(1, 0, 2, 3), (112, 16 * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)

    return torch.from_numpy(clip)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
        

#        for i, labfile in enumerate(stroke_files):     
#            k = list(self.shots[i].keys())[0] #key value for the ith dict
#            pos = self.shots[i][k]  # get list of tuples for k
#            pos.reverse()   # reverse list and keep popping
#            
#
#            # Add labels 
#            (start, end) = (-1, -1)
#            # Get the label
#            if len(pos)>0:
#                (start, end) = pos.pop()
#            # Iterate over the list of tuples and form labels for each sequence
#            for t in range(vidsSizes[i]-seq_size+1):
#                if t <= (start-seq_size):
#                    self.labels.append([0]*seq_size)   # all 0's 
#                elif t < start:
#                    self.labels.append([0]*(start-t)+[1]*(t+seq_size-start))
#                elif t <= (end+1 - seq_size):       # all 1's
#                    self.labels.append([1]*seq_size)
#                elif t <= end:
#                    self.labels.append([1]*(end+1-t) + [0]*(t+seq_size-(end+1)) )
#                else:
#                    if len(pos) > 0:
#                        (start, end) = pos.pop()
#                        if t <= (start-seq_size):
#                            self.labels.append([0]*seq_size)
#                        elif t < start:
#                            self.labels.append([0]*(start-t) + [1]*(t+seq_size-start))
#                        elif t <= (end+1 - seq_size):       # Check if more is needed
#                            self.labels.append([1]*seq_size)
#                    else:
#                        # For last part with non-action frames
#                        self.labels.append([0]*seq_size)
#                    
#            #if is_train_set:
#                # remove values with transitions eg (1, 9), (8, 2) etc
#                # Keep only (0, 10) or (10, 0) ie., single action sequences