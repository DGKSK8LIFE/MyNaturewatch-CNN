# There is a bird dataset I downloaded online for my critter directory.
# Many of the photos are in portrait mode, and when I resize them for training the proportions are really off.
# This just searches the dataset and deletes them

# Dataset was too big: deleted all files that ended in even numbers (including 0) + ended in 1 or 9

import cv2
from matplotlib.image import imread
import os
from os.path import join

path = '/home/jose/Downloads/imgs/raw-img'
for directory in os.listdir(path):
    for raw in os.listdir(join(path, directory)):
        img = imread(join(path, directory, raw))
        # Portrait picture
        if (img.shape[0] > img.shape[1]) or ((img.shape[1] / img.shape[0]) > 3):
            os.system('rm -f ' + join(path, directory, raw))