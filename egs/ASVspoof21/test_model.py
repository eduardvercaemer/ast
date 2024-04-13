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
import torchaudio


model_path = './data/models/model-5.pth'
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

#input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
# test_input = torch.rand([10, input_tdim, 128])
# test_input = test_input.to(device)
# test_output = audio_model(test_input)
# print(test_output.shape)

# spoof wav
waveform, sample_rate = torchaudio.load('./data/ASVspoof2021_DF_eval/wav/DF_E_2000011.wav')

import torchaudio.transforms as T
import torch.nn.functional as F

# Define transformation
transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=128)

# Apply transformation to waveform
spectrogram = transform(waveform)

# Interpolate the spectrogram tensor to match model's input size
spectrogram = F.interpolate(input=spectrogram.unsqueeze(0), size=[input_tdim, 128], mode='nearest')

# Adjust shape for correct model input
input_waveform = spectrogram.transpose(1, 2)

# Just take the first item of the batch
input_waveform = input_waveform[0].unsqueeze(0)

# Use the squeeze method to remove the singular dimension from the tensor
input_waveform = input_waveform.squeeze(2)

output = audio_model(input_waveform)

print(output)
