### This fie is to detect if a photo is different from sRGB
### cv2 gives a warning and there is dimension issues.

import cv2 
from matplotlib import pyplot as plt
from matplotlib.image import imread
import os
import numpy as np

# Directory of all the pictures with an animal 
critter = '/home/jose/Downloads/'

for raw in os.listdir(critter):
    image = cv2.resize(imread(critter + raw), (120, 68))
    if np.all(image.shape != (68, 120, 3)):
        print(raw)
