import numpy as np
import os
from tensorflow.keras.utils import to_categorical

def load_data(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if 'image_tile' in filename:
            img = np.load(os.path.join(directory, filename),allow_pickle=True)
            images.append(img)
        elif 'label_tile' in filename:
            lbl = np.load(os.path.join(directory, filename),allow_pickle=True)
            labels.append(lbl)
    print(len(images))
    print(len(labels))
    return np.array(images,dtype=object), np.array(labels,dtype=object)

# Load your data - update 'path_to_tiles' to your directory path
path_to_tiles = '/home/yshao/unet/data/'
images, labels = load_data(path_to_tiles)

