import pickle
import re
import shutil

import numpy as np
import os
from typing import List, Tuple, Dict, Union
import pandas as pd
import h5py
import warnings
from tqdm import tqdm

from get_recording_files import get_output_files_by_exp_group
'''
Note that this file is effectively identical to process_deg, with the exception of a few processing and extraction steps
due to the different data structures of ARBEL data compared to DEG.

The DEG specific pre-processing file logic and one-hot encoding has been substituted for steps that take in the following
table and align it with the get_recording_files structure used for the other analysis pipelines

| Pain_ML_Subject                                       | Flinching time (s)| LickingBiting time (s)| Grooming time (s)|
|-------------------------------------------------------|-------------------|-----------------------|------------------|
| capsaicin-trimmed_2024-07-15_22-04-39_1-1red-trans    | float             | float                 | float            |

where {capsaicin-trimmed_2024-07-15_22-04-39_1-1red-trans} corresponds to get_recording_files()[i][1][j]['deg']:,
'''
def parse_behavioral_summary(behavior_summary_path: os.PathLike,
                             vid_len_in_seconds=300, index_col='Pain_ML_Subject') -> Dict[str,Dict[str,float]]:
    '''

    :param behavior_summary_path: Behavior_Summary_{Project_name} output file from ARBEL_autoscore.py
    :param vid_len_in_seconds: Expecting ALL videos to be of the same length. Needed to calculate behavior frequency since
    the summary csv file does not include total video duration
    :param index_col: column name in csv file to use as key value for items in output dict

    :return:
        Dict of following type for each row of input csv

        {'index_col': {'Flinching time (s)': 0.05,
            'LickingBiting time (s)': 0.00,
            'Grooming time (s)': 0.23
        }
    '''

    # Parse csv into df
    summary_df = pd.read_csv(behavior_summary_path, index_col=0)
    time_columns = [col for col in summary_df.columns if '(s)' in col]
    # rewrite time values into frequencies
    assert isinstance(vid_len_in_seconds, (int, float)), "Ensure vid_len_in_seconds is of correct type"
    freqs_df = summary_df.copy()
    freqs_df[time_columns] = freqs_df[time_columns] / vid_len_in_seconds

    # save convert df to dict
    freq_dict = freqs_df.set_index(index_col).to_dict(orient='index')
    return freq_dict

def return_corresponding_freqs(recording_path: os.PathLike, freq_dir) -> List[float]:
    '''
    :param recording_path: Expecting in format of {recording_trans}.avi
    :param freq_dir: Dict from parse_behavioral_summary with keys in format of {recording_trans}
    :return:
        List of behavior frequencies
    '''
    recording_name = os.path.basename(recording_path).split('.')[0]
    # Check if recording_name found in freq_dir keys
    recording_freq = freq_dir.get(recording_name, False)
    if isinstance(recording_freq, Dict):
        # Process item into list
        freqs = list(recording_freq.values())
        return freqs
def generate_ARBEL_summary(data_dir: os.PathLike, ARBEL_summary_csv_path: os.PathLike) -> Tuple[str,]:
    # Load in recording filepaths
    data_by_exp_group = get_output_files_by_exp_group(data_dir)

    # import all ARBEL data
    ARBEL_freq_dict = parse_behavioral_summary(ARBEL_summary_csv_path)

    process_ARBEL_freq = []
    for exp_group_name, recordings in data_by_exp_group:
        print(f'processing {exp_group_name}...')
        # line up recording name to row in ARBEL Behavior_summary csv. Streamlines analysis and saving of summary data
        trans_paths = [recording['trans'] for recording in recordings]
        ARBEL_frequencies = [return_corresponding_freqs(file, ARBEL_freq_dict) for file in trans_paths]
        process_ARBEL_freq.append((exp_group_name, ARBEL_frequencies))
    return process_ARBEL_freq


def generate_and_save_ARBEL_summaries(data_dir: os.PathLike, target_dir: os.PathLike, behavior_summary_csv, filename='ARBEL_summaries.pkl'):
    target_path = os.path.join(target_dir, filename)
    feature_summary = generate_ARBEL_summary(data_dir, behavior_summary_csv)
    with open(target_path, 'wb') as f:
        pickle.dump(feature_summary, f)

def combine_ARBEL_outputs(output_folder: Union[os.PathLike, str], ending_motif="_ML.csv", save=False) -> pd.DataFrame:
    '''
    Takes in ARBEL_output folder path, identifies output files by ending_motif, then returns dataframe of the combined
    files
    '''
    # Get output files and sort them so that df column order is consistent
    output_files = sorted([os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith(ending_motif)])
    if not output_files:
        raise UserWarning(f'No output files found in {os.path.basename(output_folder)}. Skipping...')
    dfs = np.array([None] * len(output_files))
    for i, file in enumerate(output_files):
        # load each output file as column
        dfs[i] = pd.read_csv(file, index_col=False)
    # Ensure columns have same shape and combine into dataframe
    if len(np.unique([i.shape for i in dfs], axis=0)) == 1:
        combined = pd.concat(dfs, axis=1)
        if save:
            if not isinstance(save, str):
                save = f'{os.path.basename(output_folder)}_combined.csv'
            combined.to_csv(save)
        return combined
def main():
    data_dir = '/mnt/hd0/Pain_ML_data'
    target_dir = '/mnt/hd0/Pain_ML_data/summaries'
    ARBEL_Behavior_summary_path = '/mnt/hd0/Pain_ML_data/ARBEL/Pain_ML/ARBEL_output/Behavior_summary_Pain_ML_1205.csv'

    # Preprocessing: creates output csv and places them in the correct folder of the data_dir
    # Make sure all output .csv files are made, as they are currently stored in a probabilistic form

    # Now actually process summary
    generate_and_save_ARBEL_summaries(data_dir, target_dir, ARBEL_Behavior_summary_path)
    #combine_ARBEL_outputs('/mnt/hd0/Pain_ML_data/ARBEL/Pain_ML/ARBEL_output/capsaicin-trimmed_capsaicin_KO_0mins_batch1-2024-03-12_11-02-37-_chamber_1_-trans_Behavior')

if __name__ == "__main__":
    main()

