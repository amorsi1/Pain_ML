from unittest import TestCase

class Test(TestCase):
    def test_inside_recording_dirs(self):
        from standardize_data_filenames import process_video_directories
        data_dir = '/mnt/hd0/Pain_ML_data'
        video_directory = process_video_directories(data_dir)
        for recording, files in video_directory.items():
            print(f'recording: {recording}  |  number of files: {len(files)}')
            for file in files:
                filename = file['filename']
                if filename == 'features.h5':
                    pass
                else:
                    self.assertTrue(recording in filename, f' recording name {recording} not found in file {filename}')

