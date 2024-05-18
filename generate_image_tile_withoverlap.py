import rasterio
from rasterio.windows import Window
import numpy as np
import os

def clip_images_to_tiles_with_overlap(input_image_path, label_image_path, output_folder, tile_size=(256, 256), overlap_percent=25, pad_mode='constant'):
    """
    Clip both input and label images into smaller overlapping tiles of uniform size, padding if necessary.

    Args:
    input_image_path (str): Path to the large input image.
    label_image_path (str): Path to the large label image.
    output_folder (str): Directory where the tiles will be saved.
    tile_size (tuple): Desired size of the square tiles (height, width).
    overlap_percent (int): Percentage of each tile size to overlap.
    pad_mode (str): Padding mode as per numpy.pad (e.g., 'constant', 'reflect', etc.).
    """
    # Calculate overlap in pixels
    overlap = int(tile_size[0] * (overlap_percent / 100))
    stride = tile_size[0] - overlap

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with rasterio.open(input_image_path) as input_src, rasterio.open(label_image_path) as label_src:
        assert input_src.shape == label_src.shape, "Input and label images must be the same size."

        for j in range(0, input_src.height - overlap, stride):
            for i in range(0, input_src.width - overlap, stride):
                window = Window(i, j, tile_size[1], tile_size[0])
                
                # Handle input image
                input_data = input_src.read(window=window, boundless=True, fill_value=0)
                input_tile_filename = os.path.join(output_folder, f'input_tile_{j}_{i}.npy')
                np.save(input_tile_filename, input_data)
                
                # Handle label image
                label_data = label_src.read(window=window, boundless=True, fill_value=0)
                label_tile_filename = os.path.join(output_folder, f'label_tile_{j}_{i}.npy')
                np.save(label_tile_filename, label_data)

# Example usage
input_image_path = '0.tif'
label_image_path = 't.tif'
output_folder = '/home/yshao/unet/data'
clip_images_to_tiles_with_overlap(input_image_path, label_image_path, output_folder)
