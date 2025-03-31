import os.path
import pickle
from typing import List, Tuple, Dict

import numpy as np

from get_recording_files import get_output_files_by_exp_group
from process_ARBEL import combine_ARBEL_outputs
from process_kpms import get_top_syllables_and_viz, load_csv_column_to_list, get_top_syllables

'''
Loading Raster data 
'''
def syllable_list_to_matrix(syllable_ls: List[int], syllable_map: List[int]) -> np.ndarray:
    if not syllable_ls or not syllable_map:
        raise ValueError("Both input lists must be non-empty")
    if not all(isinstance(x, int) for x in syllable_ls + syllable_map):
        raise ValueError("All elements must be integers")

    matrix = np.zeros((len(syllable_ls), len(syllable_map)))
    for i, syllable in enumerate(syllable_map):
        matrix[:,i] = (np.array(syllable_ls) == syllable)
    return matrix

def load_kpms_raster(data_dir, top_syll_percent=90) -> List[Tuple[str,str, np.ndarray]]:
    # Initialize output list
    one_hot_syllables = []

    #Load in recording data
    data_by_exp_group = get_output_files_by_exp_group(data_dir)

    # Identify which syllables to consider
    syllable_map = sorted([int(i) for i in get_top_syllables(data_dir, top_syll_percent)])
    #Loop over all data
    for exp_group_name, recordings in data_by_exp_group:
        print(f'processing {exp_group_name}...')
        # Extract syllable information from csv
        recording_paths = [(os.path.basename(recording['trans']).split('.')[0], recording['kpms']) for recording in recordings]
        recording_syllables = [(recording_name, [int(item) for item in
                                                load_csv_column_to_list(kpms_file, 'syllable')])
                               for recording_name, kpms_file in recording_paths]

        # process syllable lists into one-hot encoded matrices with exp group tag
        syllable_matrices = [(exp_group_name, recording_name,  syllable_list_to_matrix(i, syllable_map))for
                             recording_name, i in recording_syllables]
        one_hot_syllables.extend(syllable_matrices) #TODO: consider alternative approach that doesn't use .extend

    return one_hot_syllables

def load_arbel_raster(arbel_outputs_path, to_numpy=True):
    # Get all behavior_folders
    arbel_output_folders = []
    for item in os.listdir(arbel_outputs_path):
        item_path = os.path.join(arbel_outputs_path, item)
        if os.path.isdir(item_path) and item.endswith('Behavior'):
            arbel_output_folders.append((item, item_path))
    # Extract combined table of all behaviors
    arbel_data = [None] * len(arbel_output_folders)
    for i, (folder_name, folder_path) in enumerate(arbel_output_folders):
        exp_group = folder_name.split('-')[0]
        recording_name = folder_name.split('_Behavior')[0] #split technique not necessarily needed, but will be robust against further changes
        combined_matrix = combine_ARBEL_outputs(folder_path)
        if to_numpy:
            combined_matrix = combined_matrix.to_numpy()
        arbel_data[i] = (exp_group, recording_name, combined_matrix)
    return arbel_data


def main():
    data_dir = '/mnt/hd0/Pain_ML_data'
    arbel_outputs = '/mnt/hd0/Pain_ML_data/ARBEL/Pain_ML/ARBEL_output'
    # OPTIONAL
    savepath = None #'/mnt/md0/dev/ML_Pain/notebooks'
    # savepath ='/mnt/md0/dev/ML_Pain/notebooks'

    kpms_raster = load_kpms_raster(data_dir)
    arbel_raster = load_arbel_raster(arbel_outputs)

    def combine_raster_lists(kpms_list=kpms_raster, arbel_list=arbel_raster) -> Dict[str, Dict[str, np.ndarray]]:

        def find_corresponding_item(target_fname: str, list2=arbel_list):
            for _,fname,matrix in list2:
                if fname == target_fname:
                    return matrix
            # if nothing found
            raise UserWarning(f'no corresponding matrix found for {target_fname}')
            return None

        combined_dict = {x[0]: None for x in kpms_list}
        for k, v in combined_dict.items():
            # Select only items in kpms_list that correspond to experimental group k
            exp_items = [x[1:] for x in kpms_list if x[0]==k]
            dict_of_arrays = {fname: {'kpms': kpms_array, 'arbel': find_corresponding_item(fname, arbel_list)} for fname, kpms_array in exp_items}
            combined_dict[k] = dict_of_arrays
        return combined_dict

    combined_rasters = combine_raster_lists()


    if savepath:
        print(f'saving rasters to {savepath}')
        with open(os.path.join(savepath,'kpms_raster.pkl'), 'wb') as f:
            pickle.dump(kpms_raster, f)
        with open(os.path.join(savepath,'arbel_raster.pkl'), 'wb') as f:
            pickle.dump(arbel_raster, f)
        with open(os.path.join(savepath,'combined_rasters.pkl'), 'wb') as f:
            pickle.dump(combined_rasters, f)

if __name__ == "__main__":
    main()