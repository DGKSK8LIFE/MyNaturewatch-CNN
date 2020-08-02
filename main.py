from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.model_selection import train_test_split

from matplotlib import pyplot
from matplotlib.image import imread
import os
import pandas as pd
import numpy as np



base = '/home/jose/Programming/aiml/Data/naturewatch'
critter = base + '/critter/'
no_critter = base + '/no_critter/'

def load_data():
	data = []
	labels = []
	for raw in os.listdir(critter):
		image = imread(critter + raw)
		data.append(np.array(image))
		labels.append(1)

	for raw in os.listdir(no_critter):
		# load image pixels
		image = imread(no_critter + raw)
		data.append(np.array(image))
		labels.append(0)
	data = np.array(data)
	labels = np.array(labels)
	return data, labels

data, labels = load_data()

assert data.shape[0] == labels.shape[0]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=101)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

for i, image in enumerate(X_train[:9]):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(image)
	print('image', image.shape, 'label', y_train[i])
# show the figure
pyplot.show()

model = Sequential()
# Reshape image to a much smaller size
model.add(Reshape(544, 960, 3))