import os

def rename_directories(base_dir, start_index=0):
    """
    Rename all directories in the given base directory to follow the naming convention starting from start_index, 
    preserving the order of directories as they are currently listed.
    
    :param base_dir: Path to the base directory containing the directories to rename.
    :param start_index: The starting index for renaming the directories.
    """
    print(f"Renaming directories in {base_dir}")
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    dirs.sort()  # Ensure directories are sorted before renaming
    print(f"Found directories: {dirs}")

    for idx, dirname in enumerate(dirs):
        new_name = f"{start_index + idx:03d}"
        old_path = os.path.join(base_dir, dirname)
        new_path = os.path.join(base_dir, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

def rename_files_in_directory(directory, file_extension):
    """
    Rename all files in the given directory to follow the naming convention 0000, 0001, ..., 
    preserving the order of files as they are currently listed.
    
    :param directory: Path to the directory containing the files to rename.
    :param file_extension: The file extension of the files to rename (e.g., 'png' or 'npy').
    """
    print(f"Renaming files in {directory}")
    files = [f for f in os.listdir(directory) if f.endswith(file_extension)]
    files.sort()  # Ensure files are sorted before renaming
    print(f"Found files: {files}")

    for idx, filename in enumerate(files):
        new_name = f"{idx:04d}.{file_extension}"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

def process_dataset(dataset_names, compressions):
    """
    Process all datasets by renaming directories and files according to the specified structure.
    
    :param dataset_names: List of dataset names to process (e.g., ['RealFF', 'Deepfakes', ...]).
    :param compressions: List of compression levels to process (e.g., ['c0', 'c23', 'c40']).
    """
    for dataset in dataset_names:
        for compression in compressions:
            images_dir = os.path.join(dataset, compression, 'images')
            landmarks_dir = os.path.join(dataset, compression, 'landmarks')

            if os.path.exists(images_dir):
                rename_directories(images_dir, start_index=0)
                for video_dir in sorted(os.listdir(images_dir)):
                    video_images_dir = os.path.join(images_dir, video_dir)
                    if os.path.isdir(video_images_dir):
                        rename_files_in_directory(video_images_dir, 'png')

            if os.path.exists(landmarks_dir):
                rename_directories(landmarks_dir, start_index=0)
                for video_dir in sorted(os.listdir(landmarks_dir)):
                    video_landmarks_dir = os.path.join(landmarks_dir, video_dir)
                    if os.path.isdir(video_landmarks_dir):
                        rename_files_in_directory(video_landmarks_dir, 'npy')

def main():
    dataset_names = ['RealFF', 'Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures']  # Update as needed
    compressions = ['c23']  # Update as needed
    
    process_dataset(dataset_names, compressions)

if __name__ == "__main__":
    main()
