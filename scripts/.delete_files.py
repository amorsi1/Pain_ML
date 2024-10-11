import os


def cleanup_folders(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if not (filename.endswith('ftir.avi') or filename.endswith('trans.avi')):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    folder_path = input('/mnt/hd0/Pain_ML_data/updating_old_data/reanalysis_rest').strip()

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
    else:
        confirm = input(f"Are you sure you want to delete files in {folder_path}? (yes/no): ").strip().lower()
        if confirm == 'yes':
            cleanup_folders(folder_path)
            print("Cleanup complete.")
        else:
            print("Operation cancelled.")