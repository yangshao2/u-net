import rasterio
from rasterio.windows import Window
import numpy as np

def read_and_clip_image(file_path, clip_size=(256, 256), center=(None, None)):
    """
    Read a raster file and clip it to the specified size around the center.
    
    Parameters:
    - file_path (str): Path to the raster file.
    - clip_size (tuple): Tuple of (height, width) specifying the output image size.
    - center (tuple): Tuple of (x, y) coordinates for the center of the clip. If None, clips around the center of the image.
    
    Returns:
    - numpy.ndarray: The clipped image as a numpy array with dimensions (height, width, bands).
    
    This function opens a raster file, calculates the offset needed to center the clip,
    and extracts a window of the desired size. The image data is then normalized to a range of [0, 1]
    assuming the input data range is typical of 8-bit images (0-255). If your data has a different range,
    adjust the normalization accordingly.
    """
    with rasterio.open(file_path) as src:
        width, height = src.width, src.height
        x_center, y_center = center
        if x_center is None or y_center is None:
            # Default to the center of the image if no center is specified
            x_center, y_center = width // 2, height // 2
        
        # Calculate the offset to ensure the clip is centered
        x_off = max(0, x_center - clip_size[1] // 2)
        y_off = max(0, y_center - clip_size[0] // 2)

        # Define the window of data to read based on the offset and clip size
        window = Window(x_off, y_off, clip_size[1], clip_size[0])
        clipped_image = src.read(window=window)
        
        # Normalize the image to the range [0, 1] for neural network compatibility
        clipped_image = clipped_image.astype('float32')
        clipped_image /= 255.0  # Normalization factor for 8-bit images
    
    # Reorder dimensions from (bands, height, width) to (height, width, bands) for ML compatibility
    return np.moveaxis(clipped_image, 0, -1)

def save_clipped_image(clipped_image, output_path):
    """
    Save the clipped and normalized image as a numpy array file.
    
    Parameters:
    - clipped_image (numpy.ndarray): The clipped image data.
    - output_path (str): Path where the numpy array should be saved.
    
    This function saves the preprocessed image data as a .npy file, which is a binary file format
    that stores numpy arrays. This format is efficient for loading data directly into a training pipeline.
    """
    np.save(output_path, clipped_image)

# Example usage:
file_path = '0.tif'
output_path = 'clipped_image.npy'
clipped_image = read_and_clip_image(file_path, clip_size=(256, 256))
save_clipped_image(clipped_image, output_path)
