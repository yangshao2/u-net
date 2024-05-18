import rasterio
from rasterio.windows import Window
import numpy as np
import os

def generate_tiles_with_overlap(image_path, label_path, output_dir, tile_size=(256, 256), overlap_percentage=30):
    """
    Generate overlapping tiles from given image and label (mask) files.

    Parameters:
    - image_path (str): Path to the raster file for the image.
    - label_path (str): Path to the raster file for the label (mask).
    - output_dir (str): Directory to save the tiles.
    - tile_size (tuple): Size of the tile (height, width).
    - overlap_percentage (int): Percentage of each tile that overlaps with adjacent tiles.
    """
    with rasterio.open(image_path) as img_src, rasterio.open(label_path) as lbl_src:
        assert img_src.shape == lbl_src.shape, "Image and label must be the same size"

        # Calculate the overlap in pixels
        overlap = int(tile_size[0] * (overlap_percentage / 100))
        stride_height = tile_size[0] - overlap
        stride_width = tile_size[1] - overlap

        # Create tiles using windows
        for row in range(0, img_src.height, stride_height):
            for col in range(0, img_src.width, stride_width):
                window = Window(col, row, tile_size[1], tile_size[0])
                if (row + tile_size[0] > img_src.height) or (col + tile_size[1] > img_src.width):
                    # Adjust window size at edges
                    window = Window(col, row, min(tile_size[1], img_src.width - col), min(tile_size[0], img_src.height - row))

                # Read the data in the window for both image and label
                img_data = img_src.read(window=window)
                lbl_data = lbl_src.read(window=window)

                # Normalize image data
                img_data = img_data.astype('float32') / 255.0
                img_data = np.moveaxis(img_data, 0, -1)  # from (bands, height, width) to (height, width, bands)

                # Optionally, normalize label data if necessary

                # Save each tile
                img_output_filename = f"image_tile_{row}_{col}.npy"
                lbl_output_filename = f"label_tile_{row}_{col}.npy"
                np.save(os.path.join(output_dir, img_output_filename), img_data)
                np.save(os.path.join(output_dir, lbl_output_filename), lbl_data)

# Usage
image_path = '0.tif'
label_path = 't.tif'
output_dir = '/home/yshao/unet'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
generate_tiles_with_overlap(image_path, label_path, output_dir, tile_size=(256, 256), overlap_percentage=15)
