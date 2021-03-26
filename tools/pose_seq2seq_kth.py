#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:10:39 2020

@author: arpan

@Description: Train GRU on sequence of KTH poses for action recognition.
"""
import _init_paths

import os
import torch
import numpy as np
import editdistance
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from lib.models.seq2seq import EncoderRNN, DecoderRNN, Seq2Seq
from lib.datasets.pose_dataset import PoseDataset
from lib.datasets.Video_Dataset import VideoDataset
from lib.datasets.pose_dataset import pad_collate
from utils import track_utils as utils

# Set the path of the input videos and the openpose extracted features directory
# Server Paths
DATASET = "/opt/datasets/cricket/ICC_WT20"
POSE_FEATS = "/home/arpan/cricket/output_json"
BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
hl_data_dir = "/home/arpan/VisionWorkspace/localization_rnn/numpy_vids_112x112_sc032"
#TRAIN_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/train"
#VAL_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/val"
#TEST_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/test"
#ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"

# Local Paths
if not os.path.exists(DATASET):
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    POSE_FEATS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/data/output_json"
    BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    hl_data_dir = "/home/arpan/VisionWorkspace/Cricket/localization_finetuneC3D/numpy_vids_112x112_sc032"



def train(model, optimizer, train_loader, state):
    epoch, n_epochs, train_steps = state

    losses = []
    cers = []

    # t = tqdm.tqdm(total=min(len(train_loader), train_steps))
    t = tqdm.tqdm(train_loader)
    model.train()

    for batch in t:
        t.set_description("Epoch {:.0f}/{:.0f} (train={})".format(epoch, n_epochs, model.training))
        loss, _, _, _ = model.loss(batch)
        losses.append(loss.item())
        # Reset gradients
        optimizer.zero_grad()
        # Compute gradients
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
        t.set_postfix(loss='{:05.3f}'.format(loss.item()), avg_loss='{:05.3f}'.format(np.mean(losses)))
        t.update()

    return model, optimizer
    # print(" End of training:  loss={:05.3f} , cer={:03.1f}".format(np.mean(losses), np.mean(cers)*100))


def evaluate(model, eval_loader):

    losses = []
    accs = []

    t = tqdm.tqdm(eval_loader)
    model.eval()

    with torch.no_grad():
        for batch in t:
            t.set_description(" Evaluating... (train={})".format(model.training))
            loss, logits, labels, alignments = model.loss(batch)
            preds = logits.detach().cpu().numpy()
            # acc = np.sum(np.argmax(preds, -1) == labels.detach().cpu().numpy()) / len(preds)
            acc = 100 * editdistance.eval(np.argmax(preds, -1), labels.detach().cpu().numpy()) / len(preds)
            losses.append(loss.item())
            accs.append(acc)
            t.set_postfix(avg_acc='{:05.3f}'.format(np.mean(accs)), avg_loss='{:05.3f}'.format(np.mean(losses)))
            t.update()
        align = alignments.detach().cpu().numpy()[:, :, 0]

    # Uncomment if you want to visualise weights
    # fig, ax = plt.subplots(1, 1)
    # ax.pcolormesh(align)
    # fig.savefig("data/att.png")
    print("  End of evaluation : loss {:05.3f} , acc {:03.1f}".format(np.mean(losses), np.mean(accs)))
    # return {'loss': np.mean(losses), 'cer': np.mean(accs)*100}
    
if __name__ == '__main__':
    
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
    # get list of label filenames containing temporal stroke and batsman labels
    train_lab = [f.rsplit('.',1)[0] +".json" for f in train_lst]
    val_lab = [f.rsplit('.',1)[0] +".json" for f in val_lst]
    test_lab = [f.rsplit('.',1)[0] +".json" for f in test_lst]
    train_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in train_lst]
    val_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in val_lst]
    test_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in test_lst]
    
    #####################################################################
    
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    tr_gt_batsman = [os.path.join(BAT_LABELS, f) for f in train_gt_batsman]
    sizes = [utils.getTotalFramesVid(os.path.join(DATASET, f)) for f in train_lst]
    print("Size : {}".format(sizes))
    
    bboxes = True
    epsilon = 50
    visualize = False
    n_clusters = 5
    clust_no = 1
    first_frames = 2
    
    input_size = 75
    output_size = 2
    HIDDEN_SIZE = 1024
    BATCHSIZE = 3
    N_EPOCHS = 10
    USE_CUDA = torch.cuda.is_available()
    
    tr_dataset = VideoDataset(hl_data_dir, tr_labs, sizes, seq_size=16, stride=1, is_train_set=True)

#    dataset = PoseDataset(DATASET, POSE_FEATS, train_lst, tr_labs, bboxes=False, \
#                 transforms=None)
#    eval_dataset = PoseDataset(DATASET, POSE_FEATS, val_lst, val_labs, bboxes=False, \
#                 transforms=None)
#    dataset = ToyDataset(5, 15)
#    eval_dataset = ToyDataset(5, 15, type='eval')
#    BATCHSIZE = 30
    train_loader = DataLoader(tr_dataset, batch_size=BATCHSIZE, shuffle=True)
#    eval_loader = DataLoader(eval_dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate,
#                                  drop_last=True)
    #config["batch_size"] = BATCHSIZE

    # Models
    # train the model
    model = Seq2Seq(input_size, HIDDEN_SIZE, output_size, BATCHSIZE)
    
    hidden = model.encoder.initHidden()

    if USE_CUDA:
        model = model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    print("=" * 60)
    print(model)
    print("=" * 60)

    print("\nInitializing weights...")
    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_normal_(param)

    for epoch in range(N_EPOCHS):
        run_state = (epoch, N_EPOCHS, FLAGS.train_size)

        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
        model, optimizer = train(model, optimizer, train_loader, run_state)
        evaluate(model, eval_loader)

        # TODO implement save models function
    
    
    
    # Generate examples
    f1 = torch.tensor(np.random.random((BATCHSIZE, 10, 75)), dtype=torch.float32)
    
    
    
    t1, hid = model(f1, hidden)
    