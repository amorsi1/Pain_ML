import csv
import pickle
import numpy as np
import os
from typing import List, Dict, Tuple
from get_recording_files import get_output_files_by_exp_group
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def load_csv_column_to_list(filename, column_name) -> List:
    """
    Loads specific column from a csv file, returning it as a list
    """
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the header row

        if column_name is not None:
            column_index = headers.index(column_name)
        return [row[column_index] for row in reader]

def extract_unique_syllables(list_of_kpms_filepaths: List[str]) -> Dict[str,int]:
    detected_syllables = {}
    for file in list_of_kpms_filepaths:
        # find unique vals
        recording_syllables = load_csv_column_to_list(file,'syllable')
        unique, counts = np.unique(recording_syllables, return_counts=True)
        # print(len(unique),len(counts))
        for u, c in zip(unique, counts):
            # add syllable and count if not in dict yet
            if u not in detected_syllables.keys():
                # print(f'adding syllable {u}')
                detected_syllables[u] = c
            else:
                # update count if syllable already in dict
                detected_syllables[u] += c
    return detected_syllables


def num_of_syllables_for_percentage_of_data(sorted_dict, percentage: float) -> int:
    """
    Calculates the number of syllables needed to explain a percentage of the total data

    :param sorted_dict: sorted dict of format <sorted_dict[syllable] = syllable_count> for all videos in dataset
    :param percentage: percent of data you want the syllables to explain, modulating the number of syllables returned
        by the function. Expecting values (0,100], not (0,1]

    :return: Minimum number of syllables needed to explain {percentage} percent of the data
    """
    # Calculate the total count
    total_count = sum(
        list(sorted_dict.values()))  # added list because total_count was being assigned a dict not a single number

    # Calculate cumulative sum and percentage
    cumulative_sum = np.cumsum(list(sorted_dict.values()))
    cumulative_percentage = (cumulative_sum / total_count) * 100

    return np.argmax(cumulative_percentage >= percentage) + 1

def sort_all_sylables(data_dir):
    """
    Gets all kpms output files in data_dir and sorts them by count values
    """
    # get syllable counts across all videos
    video_dir = os.path.join(data_dir, 'videos')
    kpms_filepaths = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith('kpms.csv'):
                filepath = os.path.join(root, file)
                kpms_filepaths.append(filepath)
    detected_syllables = extract_unique_syllables(kpms_filepaths)

    # sort syllable dict by count values
    sorted_dict = dict(sorted(detected_syllables.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict

def get_top_syllables(data_dir: str, percent: float) -> List[str]:
    """

    :param data_dir:
    :param percent: percent of data you want the syllables to explain, modulating the number of syllables returned by the
    function. Expecting values (0,100], not (0,1]
    :return: list of parameters that collectively make up {percent}% of the syllable labels across all videos
    """
    # sort syllable dict by count values
    sorted_dict = sort_all_sylables(data_dir)
    # select subset of syllables that meet percent criteria
    num_syllables_needed = num_of_syllables_for_percentage_of_data(sorted_dict, percent)
    top_syllables = list(sorted_dict.keys())[:num_syllables_needed]
    #TODO: change output into list of int's, and test downstream effects
    return top_syllables

def viz_top_syllables_cdf(data_dir, percent: float, save_to=None):
    sorted_dict = sort_all_sylables(data_dir)
    # Calculate the total count
    total_count = sum(list(sorted_dict.values()))
    # Calculate cumulative sum and percentage
    cumulative_sum = np.cumsum(list(sorted_dict.values()))
    cumulative_percentage = (cumulative_sum / total_count) * 100

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(sorted_dict) + 1), cumulative_percentage, marker='o')
    plt.xlabel('Number of Syllables')
    plt.ylabel('Cumulative Percentage of Total Count')
    plt.title('Cumulative Distribution of Syllable Counts')
    plt.grid(True)

    # Add reference lines
    plt.axhline(y=80, color='r', linestyle='--', label='80% Line')
    plt.axhline(y=90, color='g', linestyle='--', label='90% Line')
    # Annotate the number of syllables needed to reach 80% and 90%
    syllables_80 = np.argmax(cumulative_percentage >= 80) + 1
    syllables_90 = np.argmax(cumulative_percentage >= 90) + 1
    plt.annotate(f'80% at {syllables_80} syllables', xy=(syllables_80, 80), xytext=(syllables_80 + 1, 75),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(f'90% at {syllables_90} syllables', xy=(syllables_90, 90), xytext=(syllables_90 + 1, 85),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.legend()
    plt.tight_layout()
    # Save the figure if a path is provided
    if save_to:
        plt.savefig(save_to)
        print(f"Figure saved to {save_to}")

    plt.show()

    # Save the figure if a path is provided
    if save_to:
        plt.savefig(save_to)
        print(f"Figure saved to {save_to}")

    # Print some statistics
    print(f"Total number of syllables: {len(sorted_dict)}")
    print(f"Number of syllables needed to reach 80% of total count: {syllables_80}")
    print(f"Number of syllables needed to reach 90% of total count: {syllables_90}")
    # Print top 10 most frequent syllables
    print("\nTop 10 most frequent syllables:")
    for i, (syllable, count) in enumerate(list(sorted_dict.items())[:10], 1):
        percentage = (count / total_count) * 100
        print(f"{i}. Syllable {syllable}: {count} ({percentage:.2f}%)")

def get_top_syllables_and_viz(data_dir,percent,save_to=None):
    viz_top_syllables_cdf(data_dir,percent,save_to)
    return get_top_syllables(data_dir, percent)
def get_syllables_freqs(syllable_seq: List[str], syllable_map: List[str]):

    # Initialize freqs array in same shape as syllable_map
    freqs = np.zeros(len(syllable_map))

    # Get unique values and counts of every value in syllable_seq
    num_syllables = len(syllable_seq)
    syllables, counts = np.unique(syllable_seq, return_counts=True)

    for s, c in zip(syllables, counts):
        # if unique syllable_seq in syllable_map
        if s in syllable_map:
            index = syllable_map.index(s)
            freqs[index] = c / num_syllables
    return freqs

def generate_kpms_summary(data_dir: os.PathLike, percent: int) -> List[Tuple[str,np.ndarray]]:
    #Load in recording data
    data_by_exp_group = get_output_files_by_exp_group(data_dir)

    # Identify which syllables to consider
    syllable_map = get_top_syllables_and_viz(data_dir, percent,
                                             '/mnt/hd0/Pain_ML_data/summaries/kpms_top_syllable_selection.png')

    process_syllable_freq = []
    for exp_group_name, recordings in data_by_exp_group:
        print(f'processing {exp_group_name}...')
        kpms_paths = [recording['kpms'] for recording in recordings]
        recording_syllables = [load_csv_column_to_list(kpms_file, 'syllable') for kpms_file in kpms_paths]
        syllable_freqs = np.array([get_syllables_freqs(recording, syllable_map) for recording in recording_syllables])
        process_syllable_freq.append((exp_group_name, syllable_freqs))
    return process_syllable_freq

def generate_and_save_kpms_summary(data_dir: os.PathLike, target_dir: os.PathLike,
                                   percent: int, filename='kpms_summaries.pkl'):
    target_path = os.path.join(target_dir, filename)
    kpms_summary = generate_kpms_summary(data_dir, percent)
    with open(target_path, 'wb') as f:
        pickle.dump(kpms_summary, f)

def main():
    data_dir = '/mnt/hd0/Pain_ML_data'
    target_dir = '/mnt/hd0/Pain_ML_data/summaries'
    generate_and_save_kpms_summary(data_dir, target_dir, 90)

if __name__ == "__main__":
    main()