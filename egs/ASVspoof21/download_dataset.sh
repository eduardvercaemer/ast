#!/bin/bash
# ASVspoof21 : https://www.asvspoof.org/index2021.html
# author     : Eduard Vercaemer
#
# We only use the Deepfake dataset

set -x
wget -O- https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz | tar -C data -xzf-
wget -O- https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz?download=1 | tar -C data -xzf-
wget -O- https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part01.tar.gz?download=1 | tar -C data -xzf-
wget -O- https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part02.tar.gz?download=1 | tar -C data -xzf-
wget -O- https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part03.tar.gz?download=1 | tar -C data -xzf-
