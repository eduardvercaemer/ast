"""
Generate train and eval json files
"""
import json
import os

import numpy as np

# this is where the train.json and eval.json files will be created
spec_directory = './data/spec'
dataset_directory = './data/ASVspoof2021_DF_eval'
wav_directory = dataset_directory + '/wav'
metadata_path = './data/keys/DF/CM/trial_metadata.txt'
n_folds = 5

if __name__ == "__main__":
    if not os.path.exists(spec_directory):
        os.mkdir(spec_directory)

    print(f'generating spec files for {n_folds} folds ...')

    """
    example row:
    ['LA_0023' 'DF_E_2000011' 'nocodec' 'asvspoof' 'A14' 'spoof' 'notrim' 'progress' 'traditional_vocoder' '-' '-' '-' '-']
                audio file     codec                      tag
    """
    meta = np.loadtxt(metadata_path, delimiter=' ', dtype='str')
    meta = [(item[1], item[2], item[5]) for item in meta]
    items = len(meta)

    # partition algorithm based on esc50 dataset
    # the esc50 dataset comes already partitioned in 5 folds, each run uses 4 folds to train and the last one to
    # evaluate. since we don't have a pre-partitioned dataset, we will just divide the items in 5 folds sequentially.
    # we can improve this in the future
    for fold in range(n_folds):
        train_items = []
        eval_items = []
        for idx, item in enumerate(meta):
            label = 1 if item[2] == 'spoof' else 0

            node = {
                'wav': f'{wav_directory}/{item[0]}.wav',
                'labels': f'/m/{label}',
            }

            if idx % (n_folds + 1) == fold:
                eval_items.append(node)
            else:
                train_items.append(node)

        print(f'fold {fold}: {len(train_items)} training samples, {len(eval_items)} test samples')

        with open(f'{spec_directory}/asvspoof2021df_train_data_{fold + 1}.json', 'w') as f:
            json.dump({'data': train_items}, f, indent=1)

        with open(f'{spec_directory}/asvspoof2021df_eval_data_{fold + 1}.json', 'w') as f:
            json.dump({'data': eval_items}, f, indent=1)
