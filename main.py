import tensorflow as tf 
from matplotlib import pyplot
from matplotlib.image import imread
import os
import pandas as pd

base = '/home/jose/Programming/aiml/Data/naturewatch/critter/'

# Show first 9 images with pyplot
# having issues with cv2
for i, raw in enumerate(os.listdir(base)[:9]):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# load image pixels
	image = imread(base + raw)
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()