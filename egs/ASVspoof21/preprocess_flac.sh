#!/bin/bash
# author     : Eduard Vercaemer
#
# Resample and convert flac -> wav
# Also detect invalid audio files (from possible incomplete download) and remove them
# from dataset

set -x

mkdir -p ./data/ASVspoof2021_DF_eval/wav

find ./data/ASVspoof2021_DF_eval/flac \
  -type f \
  -print0 | \
parallel -0 -j+0 \
  'output="./data/ASVspoof2021_DF_eval/wav/{/.}.wav"; [ ! -e "$output" ] && sox {} -r 16000 "$output"; echo 1' 2>&1 | \
pv -l -s $(./data/ASVspoof2021_DF_eval/flac -type f | wc -l) > /dev/null
