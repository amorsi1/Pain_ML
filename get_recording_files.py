import os
import warnings
from typing import Dict, Tuple, List, Union


def get_recording_outputs(recording_dir: Union[str, os.PathLike]) -> Dict[str, Union[str, os.PathLike]]:
    """
    Processes a recording dir to get the path of all output files to be used for downstream analysis

    Parameters:
    recording_dir (str): The path of the recording directory. function looks for the following files:
        palmreader : filename of 'features.h5'
        kpms : ends with 'kpms.csv'
        deg : None TODO: implement naming structure
        dlc : ends with '000.h5'
        trans : ends with 'trans.avi'
        ftir: ends with 'trans.avi'

        This sort of directory structrue is achieved by running standardize_data_filenames with copy_kpms_result

    Returns:
    dict: A dictionary ...
    """

    def make_abs(path: Union[str, os.PathLike], root=recording_dir) -> Union[str, os.PathLike]:
        return os.path.join(root, path)

    keys = ['palmreader', 'kpms', 'deg', 'dlc', 'trans', 'ftir']
    output_files = {k: None for k in keys}
    output_files['filename'] = os.path.basename(recording_dir)

    for file in os.listdir(recording_dir):
        if file == 'features.h5':
            output_files['palmreader'] = make_abs(file)
        elif file.endswith('kpms.csv'):
            output_files['kpms'] = make_abs(file)
        elif file.endswith('000.h5'):
            output_files['dlc'] = make_abs(file)
        elif file.endswith('trans.avi'):
            output_files['trans'] = make_abs(file)
        elif file.endswith('ftir.avi'):
            output_files['ftir'] = make_abs(file)
    for k, v in output_files.items():
        if v is None and k != 'deg':  # TODO: remove second boolean once deg logic is implemented
            message = f'No file found for output file {k} in folder {os.path.basename(recording_dir)}'
            warnings.warn(message)

    return output_files


def get_output_files_by_exp_group(data_dir: Union[str, os.PathLike]) -> List[Tuple[str, List[Dict[str, Union[str, os.PathLike]]]]]:
    videos = os.path.join(data_dir, 'videos')
    exp_group_tuples = []

    # split by experimental group
    for exp_group in os.listdir(videos):
        exp_group_abs_path = os.path.join(videos, exp_group)
        if not os.path.isdir(exp_group_abs_path):
            continue
        recordings = os.listdir(exp_group_abs_path)
        if recordings is None:
            message = f'No files found in {exp_group}. Expecting a directory of recordings for a certain experimental group'
            warnings.warn(message)
            continue

        # get recordings within each experimental groups and append dicts of all their output file paths
        recording_output_dicts = []
        for recording in recordings:
            recording_path = os.path.join(exp_group_abs_path, recording)
            if os.path.isdir(recording_path):
                recording_output_dicts.append(get_recording_outputs(recording_path))
            else:
                print(f'argument is not a directory: recording "{recording}" in {exp_group}')
        # Append a tuple of (exp group name, list of dicts for all recordings found)
        exp_group_tuples.append((exp_group, recording_output_dicts))
    return exp_group_tuples


def main():
    data_dir = '/mnt/hd0/Pain_ML_data'
    get_output_files_by_exp_group(data_dir)


if __name__ == "__main__":
    main()
