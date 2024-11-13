import os
from typing import Dict, List

import pandas as pd
from process_features import calculate_bins, process_h5_files, calculate_frequencies
import numpy as np
from get_recording_files import get_output_files_by_exp_group
from process_kpms import get_top_syllables, load_csv_column_to_list

pain_datasets = ['paw_guarding','both_front_paws_lifted', 'luminance_logratio','hip_tailbase_hlpaw_angle',
                 'hind_paws_distance', 'neck_snout_distance']
def calculate_feature_bins(data_dir, pain_datasets) -> Dict[str,np.ndarray]:
    """Function made from process_features fn"""
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
    return bin_details


def concatenate_array_list(array_list):
    """
    Concatenate list of arrays with shape (1, ~13500, 15), truncating to shortest length
    """
    # First squeeze out the extra dimension of 1
    processed_arrays = [np.squeeze(arr, axis=0) for arr in array_list]

    # Find minimum number of rows across all arrays
    min_rows = min(arr.shape[0] for arr in processed_arrays)
    print(f"Truncating all arrays to {min_rows} rows")

    # Truncate all arrays to minimum size
    truncated_arrays = [arr[:min_rows, :] for arr in processed_arrays]

    # Concatenate horizontally
    result = np.concatenate(truncated_arrays, axis=1)

    print("Final shape:", result.shape)
    return result

def make_features_recording(data_dir, features_path: os.PathLike, pain_datasets=pain_datasets):
    # Calculate bins for feature histograms
    bin_details = calculate_feature_bins(data_dir, pain_datasets)
    # get features data specific to the experimental group
    features_data = process_h5_files([features_path], pain_datasets)

    # bin floats into ont_hot encoded arrays
    temp_list = []
    for variable, bin_edges in bin_details.items():
        # creates frequency summary representations using bins determined by looking at all data
        arr = np.array(calculate_frequencies(features_data[variable], bin_edges, return_binned_arrays=True))
        temp_list.append(arr)
    concatenated_data = concatenate_array_list(temp_list)
    one_hot_data = np.array([features_data['paw_guarding'][0], features_data['both_front_paws_lifted'][0]]).T
    return concatenated_data, one_hot_data

def one_hot_encode_kpms_syllable(syllable_data: List[str], syllable_map: List[str]):
    # make dict that maps from syllable (str) to index
    map = {s:idx for idx, s in enumerate(syllable_map)}
    # initialize empty vector
    vector = np.zeros(len(syllable_map)) #unnecessary?
    one_hot_array = np.zeros((len(syllable_data), len(syllable_map))) # num_frames x num_syllables
    for idx, syllable in enumerate(syllable_data):
        if syllable in syllable_map:
            syllable_index = map[syllable]
            one_hot_array[idx,syllable_index] = 1
        else:
            pass
    assert all([sum(i) < 2 for i in one_hot_array]), "output vector not successfully one-hot encoded"
    return one_hot_array

def make_kpms_raster(data_dir, kpms_path, percent):
    # Load in recording data
    data_by_exp_group = get_output_files_by_exp_group(data_dir)

    # Identify which syllables to consider
    syllable_map = get_top_syllables(data_dir, percent)

    # Get recording syllables
    recording_syllables = load_csv_column_to_list(kpms_path, 'syllable')
    return one_hot_encode_kpms_syllable(recording_syllables, syllable_map)

def make_deg_raster(deg_path):
    df = pd.read_csv(deg_path, index_col=0)
    # turn df into arrays with shape (num_rows, num_columns)
    return df.to_numpy()


def trim_to_smallest(arrays):
    """
    Trim all arrays to the smallest common shape.

    Parameters:
    -----------
    arrays : list of numpy.ndarray
        List of arrays to trim

    Returns:
    --------
    tuple
        (list of trimmed arrays, minimum shape)
    """
    # Find minimum shape for each dimension
    min_shape = tuple(min(dim) for dim in zip(*[arr.shape for arr in arrays]))

    # Trim each array to the minimum shape
    trimmed_arrays = [arr[tuple(slice(0, s) for s in min_shape)] for arr in arrays]

    return trimmed_arrays, min_shape


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def create_and_save_raster_animation(data, output_filename, title="Raster Plot", fps=45, window_size=100):
    """
    Create and save a raster plot animation.

    Parameters:
    -----------
    data : np.ndarray
        Binary (one-hot encoded) data array of shape (num_frames, num_columns)
    output_filename : str
        Path where to save the video (should end in .mp4)
    title : str
        Custom title for the plot
    fps : int
        Frames per second for the animation
    window_size : int
        Number of frames to show in the sliding window
    """
    # Create figure and axis with adjusted size and tight layout
    fig, ax = plt.subplots(figsize=(8, 4))  # Reduced figure size
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)  # Adjust margins

    # Get dimensions
    num_frames, num_columns = data.shape

    # Create scatter plot artists for each column
    scatters = []
    colors = plt.cm.rainbow(np.linspace(0, 1, num_columns))

    for col in range(num_columns):
        scatter = ax.scatter([], [], color=colors[col], s=10, marker='s')
        scatters.append(scatter)

    # Set up plot
    ax.set_xlim(0, window_size)
    ax.set_ylim(-0.5, num_columns - 0.5)  # Adjusted y-limits to be tighter
    ax.set_xlabel('Frame')
    ax.set_ylabel('Column')
    ax.set_title(f'{title} ({num_columns} columns)')

    def init():
        for scatter in scatters:
            scatter.set_offsets(np.empty((0, 2)))
        return scatters

    def update(frame):
        # Calculate window start
        start_idx = max(0, frame - window_size)

        # Update each scatter plot
        for col in range(num_columns):
            # Find active frames for this column in the current window
            active_frames = np.where(data[start_idx:frame, col])[0]
            if len(active_frames) > 0:
                # Convert to display coordinates
                points = np.column_stack((active_frames, np.full_like(active_frames, col)))
                scatters[col].set_offsets(points)
            else:
                scatters[col].set_offsets(np.empty((0, 2)))

        # Update x-axis limits to create sliding window effect
        ax.set_xlim(0, window_size)

        return scatters

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=num_frames,
        interval=1000/fps,
        blit=True
    )

    # Set up the writer
    writer = FFMpegWriter(
        fps=fps,
        metadata=dict(artist='Me'),
        bitrate=2000
    )

    # Save the animation
    print(f"Saving animation to {output_filename}...")
    anim.save(output_filename, writer=writer)
    plt.close()
    print("Animation saved successfully!")


# Example usage:
"""
# Create and save animation with custom title
create_and_save_raster_animation(data, 'output.mp4', title='My Custom Title')
"""


# Example usage:
"""
# Create some sample data
num_frames = 1000
num_columns = 4
data = np.zeros((num_frames, num_columns))
for col in range(num_columns):
    spike_frames = np.random.choice(num_frames, size=50, replace=False)
    data[spike_frames, col] = 1

# Save the animation
create_and_save_raster_animation(data, 'raster_animation.mp4')
"""

# Example usage:
# anim = create_raster_animation(data1, data2, data3)
# plt.show()

import cv2
import numpy as np


def combine_videos(raster_videos, square_video, output_filename, target_fps=45):
    """
    Combine three raster videos and one square video into a single synchronized video.
    Layout:
    [Square Video][Raster 1]
                  [Raster 2]
                  [Raster 3]
    """
    # Open all video captures
    raster_caps = [cv2.VideoCapture(video) for video in raster_videos]
    square_cap = cv2.VideoCapture(square_video)

    # Get video properties and print them for debugging
    width_raster = int(raster_caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height_raster = int(raster_caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    width_square = int(square_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_square = int(square_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Raster video dimensions: {width_raster}x{height_raster}")
    print(f"Square video dimensions: {width_square}x{height_square}")

    # Calculate the target dimensions for the square video
    # It should be the same height as three stacked raster plots
    target_square_height = 3 * height_raster
    target_square_width = int(width_square * (target_square_height / height_square))

    # Calculate output dimensions
    output_width = target_square_width + width_raster
    output_height = target_square_height

    print(f"Output video dimensions: {output_width}x{output_height}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_filename,
        fourcc,
        target_fps,
        (output_width, output_height)
    )

    print("Starting video combination...")
    frame_count = 0

    while True:
        # Read frames from all videos
        raster_frames = []
        for cap in raster_caps:
            ret, frame = cap.read()
            if not ret:
                break
            # Ensure raster frame is the correct size
            if frame.shape[0] != height_raster or frame.shape[1] != width_raster:
                frame = cv2.resize(frame, (width_raster, height_raster))
            raster_frames.append(frame)

        ret, square_frame = square_cap.read()

        # Check if any video has ended
        if not ret or len(raster_frames) != len(raster_caps):
            break

        # Resize square frame
        square_frame = cv2.resize(square_frame, (target_square_width, target_square_height))

        # Create the combined frame
        combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)

        # Place square video on the left
        combined_frame[:, :target_square_width] = square_frame

        # Place raster videos stacked vertically on the right
        for i, frame in enumerate(raster_frames):
            y_start = i * height_raster
            y_end = y_start + height_raster
            x_start = target_square_width
            x_end = output_width

            try:
                combined_frame[y_start:y_end, x_start:x_end] = frame
            except ValueError as e:
                print(f"Error at frame {frame_count}, position {i}")
                print(f"Combined frame shape: {combined_frame.shape}")
                print(f"Target region shape: {y_end - y_start}x{x_end - x_start}")
                print(f"Source frame shape: {frame.shape}")
                raise e

        # Write the combined frame
        out.write(combined_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    # Release everything
    for cap in raster_caps:
        cap.release()
    square_cap.release()
    out.release()

    print(f"Video combination complete! Output saved to {output_filename}")
    print(f"Total frames processed: {frame_count}")


def plot_onehot_histogram(data, save_path=None, plot_title=None, figsize=(10, 6), dpi=300):
    """
    Create a histogram showing the sum of values for each column in a one-hot encoded array.

    Parameters:
    -----------
    data : numpy.ndarray
        One-hot encoded array of shape (13500, n)
    save_path : str, optional
        Full path where to save the plot. If None, plot is only displayed.
    plot_title : str, optional
        Custom title for the plot. If None, uses default title.
    figsize : tuple, optional
        Figure size (width, height)
    dpi : int, optional
        Resolution for saved figure

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    # Sum values in each column
    column_sums = np.sum(data, axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot
    plt.bar(range(len(column_sums)), column_sums,
            edgecolor='black', alpha=0.7, color='skyblue')

    # Add summary statistics
    stats_text = f'Mean count: {np.mean(column_sums):.2f}\n'
    stats_text += f'Std dev: {np.std(column_sums):.2f}\n'
    stats_text += f'Median count: {np.median(column_sums):.2f}\n'
    stats_text += f'Total columns: {len(column_sums)}'

    plt.text(0.95, 0.95, stats_text,
             transform=ax.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel('Column Index')
    plt.ylabel('Count (sum of ones)')

    # Set custom title or default
    title = plot_title if plot_title is not None else 'Distribution of Counts Across Columns'
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save with specified dpi
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig, ax


def main():
    data_dir = '/mnt/hd0/Pain_ML_data'
    target_dir = '/mnt/hd0/Pain_ML_data/summaries'

    # Select random formalin video
    data_by_exp_group = get_output_files_by_exp_group(data_dir)
    formalin_recordings = [i[1][0] for i in data_by_exp_group if i[0] == 'formalin']
    selected_vid = formalin_recordings[0] #select first recording dict
    features_raster, features_one_hot = make_features_recording(data_dir, selected_vid['palmreader'])
    plot_onehot_histogram(features_raster, save_path='figs/features.png', plot_title='Palmreader')


    kpms_raster = make_kpms_raster(data_dir, selected_vid['kpms'], 94)
    plot_onehot_histogram(kpms_raster, save_path='figs/kpms.png', plot_title='Keypoint-MoSeq')

    pain_behavior_raster = make_deg_raster(selected_vid['deg'])
    pain_behavior_raster = np.concatenate([pain_behavior_raster, features_one_hot[:13488]], axis=1)
    plot_onehot_histogram(pain_behavior_raster, save_path='figs/deg.png', plot_title='Pain Behaviors')

    # create video
    # Create and show animation
    create_and_save_raster_animation(kpms_raster, os.path.join(target_dir, 'kpms.mp4'), title='Keypoint-Moseq')
    create_and_save_raster_animation(pain_behavior_raster, os.path.join(target_dir, 'deg.mp4'), title='Pain Behaviors')
    create_and_save_raster_animation(features_raster, os.path.join(target_dir, 'features.mp4'), title='Palmreader')


    # Combine videos
    raster_videos = [
        os.path.join(target_dir,'features.mp4'),
        os.path.join(target_dir,'deg.mp4'),
        os.path.join(target_dir,'kpms.mp4')
    ]
    square_video = selected_vid['trans']
    combine_videos(raster_videos, square_video, os.path.join(target_dir,'combined_raster.mp4'))

if __name__ == "__main__":
    main()