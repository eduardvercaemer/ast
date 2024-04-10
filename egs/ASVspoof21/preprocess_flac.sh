#!/bin/bash
# author     : Eduard Vercaemer
#
# Resample and convert flac -> wav
# Also detect invalid audio files (from possible incomplete download) and remove them
# from dataset

set -x
mkdir -p ./data/ASVspoof2021_DF_eval/wav
find ./data/ASVspoof2021_DF_eval/flac -type f -name '*.flac' \
  -exec sh -c 'sox "$0" -r 16000 "./data/ASVspoof2021_DF_eval/wav/$(basename ${0%.flac}).wav"' {} \;
