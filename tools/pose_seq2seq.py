#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:13:36 2020

@author: arpan
"""
import _init_paths

import torch
import numpy as np
import editdistance
import tqdm
from matplotlib import pyplot as plt
from lib.models import EncoderRNN, DecoderRNN, Seq2Seq
from lib.datasets import PoseDataset



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
    input_size = 75
    output_size = 2
    HIDDEN_SIZE = 1024
    BATCHSIZE = 3
    N_EPOCHS = 10
    USE_CUDA = torch.cuda.is_available()

    dataset = PoseDataset()
#    dataset = ToyDataset(5, 15)
#    eval_dataset = ToyDataset(5, 15, type='eval')
#    BATCHSIZE = 30
    train_loader = data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate, drop_last=True)
    eval_loader = data.DataLoader(eval_dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate,
                                  drop_last=True)
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

    for epoch in range(n_epochs):
        run_state = (epoch, N_EPOCHS, FLAGS.train_size)

        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
        model, optimizer = train(model, optimizer, train_loader, run_state)
        evaluate(model, eval_loader)

        # TODO implement save models function
    
    
    
    # Generate examples
    f1 = torch.tensor(np.random.random((BATCHSIZE, 10, 75)), dtype=torch.float32)
    
    
    
    t1, hid = model(f1, hidden)
    