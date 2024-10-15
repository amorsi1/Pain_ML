import pickle

import numpy as np
import os
from typing import List, Dict, Tuple
import h5py
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from get_recording_files import get_output_files_by_exp_group
import logging
import warnings


pain_datasets = ['paw_guarding','both_front_paws_lifted', 'luminance_logratio','hip_tailbase_hlpaw_angle',
                 'hind_paws_distance', 'neck_snout_distance']

def extract_datasets_from_h5(filepath: os.PathLike, target_datasets: List[str]) -> Dict[str, np.ndarray]:
    assert filepath.endswith('.h5'), f' Expecting h5 file, but file {os.path.basename(filepath)} passed in'
    with h5py.File(filepath, 'r') as f:
        # Get top level groups
        groups = [key for key in f.keys() if isinstance(f[key], h5py.Group)]

        # Make sure there is only one group
        if len(groups) != 1:
            raise ValueError(
                f' expecting file to have exactly one group, as is typical in palmreader outputs, but found groups: {groups}')
        group = f[groups[0]]

        # Get datasets in the group
        extracted_datasets = {}
        for dataset in target_datasets:
            if dataset not in group:
                raise KeyError(f"Dataset {dataset} not found in the file. Found the following keys: {group.keys()}")
            # Read only the desired dataset into memory and assign to a dict
            extracted_datasets[dataset] = group[dataset][()]

        return extracted_datasets


def process_single_file(args):
    filepath, pain_datasets = args
    return extract_datasets_from_h5(filepath, pain_datasets)


def process_h5_files(file_list: List[str], pain_datasets: List[str]) -> Dict[str, List[np.ndarray]]:
    # Validate input files
    valid_files = [file for file in file_list if os.path.exists(file) and file.endswith('.h5')]

    if not valid_files:
        raise ValueError("No valid .h5 files provided.")

    # Initialize dictionary to store data
    all_data = {dataset: [] for dataset in pain_datasets}

    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(process_single_file, [(f, pain_datasets) for f in valid_files]), total=len(valid_files)))

    # Collect results
    for i, result in enumerate(results):
        for dataset in pain_datasets:
            if dataset in result:
                all_data[dataset].append(result[dataset])
            else:
                print(f"Warning: Dataset {dataset} not found in file {valid_files[i]}")
                all_data[dataset].append(np.array([]))  # Append an empty array for missing data

    return all_data


def extract_datasets_from_h5(filepath: os.PathLike, target_datasets: List[str]) -> Dict[str, np.ndarray]:
    with h5py.File(filepath, 'r') as f:
        group = list(f.keys())[0]
        return {dataset: f[group][dataset][()] for dataset in target_datasets if dataset in f[group]}


def calculate_bins(arrays: List[np.ndarray], num_bins: int) -> np.ndarray:
    """
    Calculate bin edges based on the combined data from all input arrays.

    :param arrays: List of NumPy arrays containing the data
    :param num_bins: Number of bins to create
    :return: NumPy array of bin edges
    """
    # Combine all arrays
    combined = np.concatenate(arrays)

    # Calculate bin edges
    _, bin_edges = np.histogram(combined, bins=num_bins, range=(np.nanmin(combined), np.nanmax(combined)))

    return bin_edges


def calculate_frequencies(arrays: List[np.ndarray], bin_edges: np.ndarray) -> np.ndarray:
    """
    Calculate frequencies for each array based on the given bin edges.

    :param arrays: List of NumPy arrays containing the data
    :param bin_edges: NumPy array of bin edges
    :return: 2D NumPy array of frequencies for each input array
    """
    frequencies = []
    n_bins = len(bin_edges) - 1
    for i, arr in enumerate(arrays):
        # Use numpy's digitize to find which bin each value belongs to
        # indices = np.digitize(arr, bin_edges[1:])
        # Count occurrences in each bin
        # bin_counts = np.bincount(indices, minlength=len(bin_edges) - 1)
        # # Calculate frequencies
        # freq = bin_counts / len(arr)
        # frequencies.append(freq)

        hist, _ = np.histogram(arr, bins=bin_edges, density=True)

        if len(hist) != n_bins:
            warnings.warn(f"Warning: Histogram for array {i} has {len(hist)} bins instead of the expected {n_bins}")

        frequencies.append(hist)
    return frequencies



def concatenate_arrays(list_of_lists: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    This function takes a list of lists, where each inner list contains NumPy arrays.
    It concatenates the arrays that are in the same position across all inner lists.

    :param list_of_lists (List[List[np.ndarray]]): A list of lists containing NumPy arrays
        Each inner list represents a group, and arrays in the same position
        across groups will be concatenated.

    :return List[np.ndarray]: A list of concatenated NumPy arrays. The length of this list
        will be equal to the length of the longest inner list in the input.

    Example:
        >>> input_list = [
        ...     [np.array([1, 2])],
        ...     [np.array([3, 4])],
        ...     [np.array([5, 6])]
        ... ]
        >>> result = concatenate_arrays(input_list)
        >>> print(result)
        [array([1, 2, 3, 4, 5, 6])]
    """
    # Transpose the list of lists to group arrays by position
    transposed = list(map(list, zip(*list_of_lists)))

    # Concatenate arrays for each position, ignoring None values
    result = [np.concatenate([arr for arr in arrays if arr is not None]) for arrays in transposed]

    return result

def process_one_hot_datasets(arrays: List[np.ndarray]) -> List[float]:
    int_one_hot = [arr.astype(int) for arr in arrays]
    freqs = [np.sum(arr)/len(arr) for arr in int_one_hot]
    return freqs

def generate_feature_summaries(data_dir: os.PathLike) -> List[Tuple[str,List[np.ndarray],List[Tuple[float,float]]]]:
    """
    Generates summaries of data in features.h5 files to be used as inputs for a linear classifier.

    Float-type data is first collected from all recordings in data_dir (agnostic of experimental group), then bins are
    fit to the concatenated data. These bins are then used to summarize individual recordings based on their histogram
    profile by concatenating all histograms together for a given recordings

    one hot encoded data is summarized simply by their frequency, and saved separately to the float-type summary

    :param data_dir: root of experiment
    :return:
        List of (group_name, float_summary, one_hot_summary) tuples where:
        - group_name (str): Experimental group name (from parent directory)
        - float_summary (List[np.ndarray]): List of float summary of recordings, with each recording represented by an array:
        - one_hot_summary (Tuple[float,float]): Tuple of frequencies of one hot encoded data for each recording. Currently
            only 'paw_guarding' and 'both_front_paws_lifted' included
    """
    # Get all features.h5 files regardless of experimental group
    # data_dir = '/mnt/hd0/Pain_ML_data'
    video_dir = os.path.join(data_dir, 'videos')

    features_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file == 'features.h5':
                filepath = os.path.join(root, file)
                features_files.append(filepath)
    print('number of files found:', len(features_files))

    # get data of pain_associated features
    concatenated_features = process_h5_files(features_files, pain_datasets)
    # use all concatenated data to calculate bins

    # bin_edges = {}
    # for dataset, data in concatenated_features.items():
    #     bin_edges[dataset] = calculate_bins(data, 15)
    bin_details = {
        # 'paw_guarding': calculate_bins(concatenated_features['paw_guarding'],15),
        # 'both_front_paws_lifted': calculate_bins(concatenated_features['both_front_paws_lifted'], 15),
        'luminance_logratio': calculate_bins(concatenated_features['luminance_logratio'], 15),
        'hip_tailbase_hlpaw_angle': calculate_bins(concatenated_features['hip_tailbase_hlpaw_angle'], 15),
        'hind_paws_distance': calculate_bins(concatenated_features['hind_paws_distance'], 15),
        'neck_snout_distance': calculate_bins(concatenated_features['neck_snout_distance'], 15),
    }

    # Now that bins are calculated, can convert timeseries data into a histogram summary representation

    data_by_exp_group = get_output_files_by_exp_group(data_dir)
    processed_summaries = []
    for exp_group_name, recordings in data_by_exp_group:
        print(f'processing {exp_group_name}...')
        # get features data specific to the experimental group
        features_paths = [recording['palmreader'] for recording in recordings]
        features_data = process_h5_files(features_paths, pain_datasets)

        # process binned summaries
        temp_list = []
        for variable, bin_edges in bin_details.items():
            # creates frequency summary representations using bins determined by looking at all data
            temp_list.append(calculate_frequencies(features_data[variable], bin_edges))
        concatenated_data = concatenate_arrays(temp_list)

        # process (boolean) one-hot encoded datasets
        paw_guard = process_one_hot_datasets(features_data['paw_guarding'])
        paws_lifted = process_one_hot_datasets(features_data['both_front_paws_lifted'])
        one_hot_data = list(zip(paw_guard, paws_lifted))
        processed_summaries.append((exp_group_name, concatenated_data, one_hot_data))
    return processed_summaries

def generate_and_save_feature_summaries(data_dir: os.PathLike, target_dir: os.PathLike, filename='palmreader_features_summaries.pkl'):
    target_path = os.path.join(target_dir, filename)
    feature_summary = generate_feature_summaries(data_dir)
    with open(target_path, 'wb') as f:
        pickle.dump(feature_summary, f)

def main():
    data_dir = '/mnt/hd0/Pain_ML_data'
    target_dir = '/mnt/hd0/Pain_ML_data/summaries'
    generate_and_save_feature_summaries(data_dir, target_dir)


if __name__ == "__main__":
    main()