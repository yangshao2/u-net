import os
import numpy as np

def print_npy_file_shapes(directory):
    """
    Prints the shapes of all .npy files in the specified directory.

    Parameters:
        directory (str): The path to the directory containing .npy files.
    """
    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an .npy file
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            # Load the .npy file
            data = np.load(file_path)
            # Print the shape of the numpy array
            print(f"{filename}: {data.shape}")

# Usage example
directory_path = '/home/yshao/unet/data'  # Replace with the path to your directory
print_npy_file_shapes(directory_path)
