import os
import warnings
import shutil
from typing import Dict
import re

def process_video_directories(project_root):
    """
    Recursively processes all video directories within the project structure,
    collecting detailed information about video files in each directory.

    This function traverses the directory structure starting from the 'videos'
    folder in the project root. It identifies directories containing video files
    (determined by common video file extensions) and collects information about
    all files in these directories.

    Parameters:
    project_root (str): The root directory of the project containing the 'videos' folder.

    Returns:
    dict: A dictionary where each key is a video directory name and each value is a list
          of dictionaries containing information about the files in that directory.
          The structure is as follows:
          {
              'video_dir_name': [
                  {
                      'filename': str,
                      'full_path': str,
                      'extension': str
                  },
                  ...
              ],
              ...
          }
    """
    videos_root = os.path.join(project_root, "videos")
    video_directories = {}

    for root, dirs, files in os.walk(videos_root):
        # Check if this is a video directory (contains video files)
        if any(file.endswith(('.avi', '.mp4', '.mov')) for file in files):
            video_name = os.path.basename(root)
            video_files = []

            try:
                for filename in files:
                    full_path = os.path.join(root, filename)
                    video_files.append({
                        'filename': filename,
                        'full_path': full_path,
                        'extension': os.path.splitext(filename)[1]
                    })

                if not video_files:
                    print(f"The directory {root} is empty.")
                else:
                    video_directories[video_name] = video_files

            except Exception as e:
                print(f"An error occurred processing {root}: {e}")

    return video_directories

def flexi_add_exp_name(recording_name, exp_group):
    # check if recording_name is already in the format {exp_group}-{recording}-{suffix}
    match = re.match(r'^(.*?)-', recording_name)
    if match:
        text_before_dash = match.group(1)
        if text_before_dash == exp_group:
            return recording_name
        else:
            return f'{exp_group}-{recording_name}'
    return f'{exp_group}-{recording_name}'

def process_recording(recording: str, files: list):
    recording_path = os.path.dirname(files[0]['full_path'])
    exp_group = os.path.basename(os.path.dirname(recording_path))

    # add experimental group to recording name
    exp_group_recording = flexi_add_exp_name(recording,exp_group)

    replacement_trans_string = f'{exp_group_recording}-trans'
    new_ftir_fname = f'{exp_group_recording}-ftir.avi'

    def replace_string_and_rename(file: Dict[str,str],old_name: str,new_name: str):
        """
        Checks if given file contains the old name. If it does then it replaces part of the filename then renames the
        files with the new naming structure implemented
        """
        if old_name in file['filename']:
            new_name = file['filename'].replace(old_name, new_name)
            os.rename(file['full_path'],
                      os.path.join(recording_path, new_name))

    # if file is from analysis-public processing pipeline
    # files saved as 'ftir_resize.avi', 'trans_resize.avi','trans_resizeDLC....h5'
    if any('ftir_resize.avi' in filename for filename in os.listdir(recording_path)):
        print(f'Processing recording {recording}')
        # rename ftir file
        new_ftir_fpath = os.path.join(recording_path, new_ftir_fname)
        os.rename(os.path.join(recording_path, 'ftir_resize.avi'), new_ftir_fpath)
        # rename any files with 'trans_resize' in them

        for file in files:
            replace_string_and_rename(file, 'trans_resize', replacement_trans_string)

    # if file is from older version of analysis-public
    # files saved as '{name_format}_body.avi', 'trans_resize.avi','trans_resizeDLC....h5'
    if any(file['filename'].endswith('body.avi') for file in files):
        print(f'Processing recording {recording}')
        # get old naming structure
        body_filename_format = None
        for file in files:
            filename = file['filename']
            if filename.endswith('_body.avi'):
                body_filename_format = os.path.splitext(filename)[0]
        if body_filename_format:
            # rename ftir
            ftir_file_path = os.path.join(recording_path, f'{body_filename_format[:-4]}ftir.avi')
            if os.path.exists(ftir_file_path):
                os.rename(ftir_file_path, os.path.join(recording_path, new_ftir_fname))
            # rename rest of the files (i.e. the 'body' files)
            for file in files:
                replace_string_and_rename(file, body_filename_format, replacement_trans_string)
        else:
            message = f'''
                older file format found in recording {recording} but no common naming format was found
    
                Directory should look similar to this
    
                ├── videos/
                │   ├── exp_group_1/
                │   │   ├── recording
                │   │   │   ├── trimmed_SN_grp2_0mins-2024-02-07_11-08-27-_chamber_2_body.avi
                │   │   │   ├── trimmed_SN_grp2_0mins-2024-02-07_11-08-27-_chamber_2_ftir.avi
                │   │   │   ├── trimmed_SN_grp2_0mins-2024-02-07_11-08-27-_chamber_2_bodyDLC_resnet50_arcteryx500Nov4shuffle1_350000.h5
    
                But instead has the following file names: {[print(file['filename']) for file in files]}
                '''
            warnings.warn(message, UserWarning)

def text_before_dash(text: str) -> str:
    match = re.match(r'^(.*?)-', text)
    if match:
        return match.group(1)

def append_to_filename(filename: str, string_to_append: str) -> str:
    name, ext = os.path.splitext(filename)
    return f"{name}{string_to_append}{ext}"

def copy_kpms_results(results_dir, data_dir):
    """
    Makes copy of files from results_path directory in data_dir where the {recording} name in the results_path dir
    is used to put the results in the correct destination directory

    :param results_dir:dir path of kpms output, found in project_dir > model > result
        Files are expected to be of the format {exp_group}-{recording_name}DLCblahblah.csv

    :param data_dir: target directory path of the following format
    ├── videos/
        ├── exp_group_1/
        │   ├── recording
        │   │   ├── recording-trans.avi
    """
    results = [file for file in os.listdir(results_dir) if file.endswith('.csv')]

    # Create regex pattern to extract exp-group (everything before the dash) and the recording name (everything between the dash and DLC suffix)
    pattern = r'^(.*?)-(.*?)-transDLC'
    # Iterate through files in output directory
    for filename in results:
        match = re.match(pattern, filename)
        if match:
            exp_group, unique_recording = match.groups()

            # Construct destination path
            dest_path = os.path.join(data_dir, 'videos', exp_group, unique_recording)

            if not os.path.exists(dest_path):
                # recording folder name may be {exp_group}-{unique_recording}
                dest_path_retry = os.path.join(data_dir, 'videos', exp_group, f'{exp_group}-{unique_recording}')
                if os.path.exists(dest_path_retry):
                    dest_path = dest_path_retry
                    pass
                else:
                    message = f"""
                    Skipping file {filename}
                    No directory found at destination path {dest_path} 
                    Or at destination path {dest_path_retry}
                    Ensure filename and destination directory match
                    """
                    warnings.warn(message, RuntimeWarning)
                    continue

            # Move the file
            src_file = os.path.join(results_dir, filename)
            dest_file = os.path.join(dest_path, append_to_filename(filename,'-kpms'))
            shutil.copy2(src_file, dest_file)
            print(f"Moved {filename} to {dest_path}")
        else:
            print(f"Skipped {filename} - doesn't match expected filename format")

def main():
    data_dir = '/mnt/hd0/Pain_ML_data'
    results_dir = r'/mnt/hd0/keypoint-moseq/formalin_capsaicin_PBS/FCP_project/2024_09_23-00_12_00/results'
    video_directory = process_video_directories(data_dir)

    for recording, files in video_directory.items():
        process_recording(recording,files)

    copy_kpms_results(results_dir, data_dir)

if __name__ == "__main__":
    main()



