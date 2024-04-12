# -*- coding: utf-8 -*-
# @Time    : 6/28/21 4:54 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : load_pretrained_model.py

# sample code of loading a pretrained AST model

import os, sys
parentdir = str(os.path.abspath(os.path.join(__file__ ,"../../../")))+'/src'
sys.path.append(parentdir)

import models
import torch

model_path = './exp/test-asvspoof21-f10-t10-impTrue-aspTrue-b32-lr1e-5/fold5/models/best_audio_model.pth'
input_tdim = 512 # same as run_asvspoof21.sh audio_length

# initialize an AST model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(model_path, map_location=device)
audio_model = models.ASTModel(
    label_dim=2, # number of tags same as n_class
    fstride=10,
    tstride=10,
    input_fdim=128,
    input_tdim=input_tdim,
    imagenet_pretrain=True,
    audioset_pretrain=True,
    model_size='base384')

audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd, strict=False)

# input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
test_input = torch.rand([10, input_tdim, 128])
test_output = audio_model(test_input)
print(test_output.shape)
