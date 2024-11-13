import pickle
import re
import shutil

import numpy as np
import os
from typing import List, Tuple
import pandas as pd
import h5py
import warnings
from tqdm import tqdm

from get_recording_files import get_output_files_by_exp_group


def get_deg_files(deg_DATA_dir: os.PathLike) -> List[os.PathLike]:
    """Extract filepaths of _output.h5 files in DATA directory of the DeepEthogram project folder"""
    deg_files = []
    for root, dirs, files in os.walk(deg_DATA_dir):
        for file in files:
            if file.endswith('_outputs.h5'):
                filepath = os.path.join(root, file)
                deg_files.append(filepath)
    print('number of files found:', len(deg_files))

    # Intermediary step to get rid of any files used for training but not relevant to this analysis
    def filter_files(files, allowed_prefixes):
        print(f'Filtering filepaths based on the following prefixes: {allowed_prefixes}')
        filtered_files = []
        for file in files:
            # Get the base name of the directory (last part of the path)
            base_name = os.path.basename(file)
            # Check if the base name contains a dash
            if '-' in base_name:
                # Get the part before the dash
                prefix = base_name.split('-')[0].lower()
                # Check if the prefix is in the allowed list
                if prefix in allowed_prefixes:
                    filtered_files.append(file)
        return filtered_files

    allowed_prefixes = ['formalin', 'pbs', 'capsaicin']
    filtered_deg_files = filter_files(deg_files, allowed_prefixes)
    print('filtered number of files', len(filtered_deg_files))
    deg_files = filtered_deg_files
    return deg_files


def get_one_hot_predictions(deg_path: str) -> pd.DataFrame:
    with h5py.File(deg_path, 'r') as f:
        # Get top level groups
        groups = [key for key in f.keys() if isinstance(f[key], h5py.Group)]
        datasets = f['resnet18']

        # Get relevant datasets
        classes = datasets['class_names'][:]
        p = datasets['P'][:]
        thresholds = datasets['thresholds'][:]

        # Construct df from values
        df_dict = {}
        for idx, (threshold, class_name) in enumerate(zip(thresholds, classes)):
            class_name = class_name.decode('UTF-8')  # without this step the value will be b'{class_name}'
            probabilities = p[:, idx]
            # one_hot encode using threshold and probabilities
            one_hot = np.where(probabilities > threshold, 1, 0)
            df_dict[class_name] = one_hot
        df = pd.DataFrame.from_dict(df_dict)
        # print('data sums')
        # print(df.sum())
        return df


def save_predictions_as_csv(deg_path, save_path):
    df = get_one_hot_predictions(deg_path)
    df.to_csv(save_path)

def generate_deg_output_and_save(deg_DATA_dir: os.PathLike, overwrite=False):
    deg_files = get_deg_files(deg_DATA_dir)
    save_paths = [i[:-3] +'_binary.csv' for i in deg_files]
    for deg_file, save_path in tqdm(zip(deg_files, save_paths), total=len(deg_files), desc="Processing files"):
        if os.path.exists(save_path) and not overwrite:
            tqdm.write(f'csv file already found for {os.path.basename(deg_file)}! Skipping...')
            continue
        try:
            save_predictions_as_csv(deg_file, save_path)
        except PermissionError as err:
            warnings.warn("""
            DeepEthogram changes permission of files in DATA folder to admin only. Make sure your user is admin or 
            change permissions so that all users can write files in the folder
            
            Useful command-line solutions:
            
            sudo chown -R yourusername:yourusername /project_root/DATA
                changes owner of the directory to your current user to give you full read, write, execute access
            
            sudo chmod -R 777 /project_root/DATA
                changes permissions so all users can write into this file. Generally considered not best practice, but oh well!
            """)
            raise
    return save_paths

def copy_deg_results(deg_paths: List[os.PathLike], data_dir = os.PathLike):
    # Create regex pattern to extract exp-group (everything before the dash) and the recording name (everything between the dash and DLC suffix)
    pattern = r'^(.*?)-(.*?)-centered'
    # Iterate through files in output directory
    for src_file in tqdm(deg_paths):
        filename = os.path.basename(src_file)
        match = re.match(pattern, filename)
        if match:
            exp_group, unique_recording = match.groups()

            # Construct destination path
            dest_dir_path = os.path.join(data_dir, 'videos', exp_group, unique_recording)

            if not os.path.exists(dest_dir_path):
                # recording folder name may be {exp_group}-{unique_recording}
                dest_path_retry = os.path.join(data_dir, 'videos', exp_group, f'{exp_group}-{unique_recording}')
                if os.path.exists(dest_path_retry):
                    # try different path format
                    dest_dir_path = dest_path_retry
                    pass
                else:
                    message = f"""
                        Skipping file {filename}
                        No directory found at destination path {dest_dir_path} 
                        Or at destination path {dest_path_retry}
                        Ensure filename and destination directory match
                        """
                    warnings.warn(message, RuntimeWarning)
                    continue

            # Move the file

            dest_file = os.path.join(dest_dir_path, f'{exp_group}-{unique_recording}-deg.csv')
            # note: this will overwrite the file at the destination path if one already exists
            shutil.copy2(src_file, dest_file)
            # print(f"Moved {filename} to {dest_dir_path}")
        else:
            print(f"Skipped {filename} - doesn't match expected filename format")

def summarize_deg_csv(csv_path) -> List[float]:
    df = pd.read_csv(csv_path, index_col=0)
    frequencies = (df.sum() / len(df)).tolist()
    return frequencies

def generate_deg_summary(data_dir: os.PathLike) -> Tuple[str,]:
    # Load in recording data
    data_by_exp_group = get_output_files_by_exp_group(data_dir)

    process_deg_freq = []
    for exp_group_name, recordings in data_by_exp_group:
        print(f'processing {exp_group_name}...')
        deg_paths = [recording['deg'] for recording in recordings]
        deg_frequencies = [summarize_deg_csv(file) for file in deg_paths]
        process_deg_freq.append((exp_group_name, deg_frequencies))
    return process_deg_freq

def generate_and_save_deg_summaries(data_dir: os.PathLike, target_dir: os.PathLike, filename='deg_summaries.pkl'):
    target_path = os.path.join(target_dir, filename)
    feature_summary = generate_deg_summary(data_dir)
    with open(target_path, 'wb') as f:
        pickle.dump(feature_summary, f)

def main():
    data_dir = '/mnt/hd0/Pain_ML_data'
    target_dir = '/mnt/hd0/Pain_ML_data/summaries'

    # Preprocessing: creates output csv and places them in the correct folder of the data_dir
    deg_DATA_root = '/mnt/md0/DEG_behavioral_labelling/08.24_behavior_deepethogram/DATA'
    # Make sure all output .csv files are made, as they are currently stored in a probabilistic form
    deg_csv_paths = generate_deg_output_and_save(deg_DATA_root)
    copy_deg_results(deg_csv_paths, data_dir)

    # Now actually process summary
    generate_and_save_deg_summaries(data_dir, target_dir)

if __name__ == "__main__":
    main()

