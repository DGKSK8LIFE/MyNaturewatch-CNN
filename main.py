from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from matplotlib import pyplot
from matplotlib.image import imread
import tensorflow as tf
import os
import numpy as np
import cv2

base = '/home/jose/Programming/aiml/Data/naturewatch'
# Directory of all the pictures with an animal 
critter = base + '/critter/'
# Directory of all the pictures without an animal
no_critter = base + '/no_critter/'

def load_data():
	data = []
	labels = []
	for raw in os.listdir(critter):
		# The array of values
		image = cv2.resize(imread(critter + raw), (120, 68))
		print(image.shape)
		data.append(np.array(image))
		# 1 for yes critter
		labels.append(np.array([0, 1]))
		# image.shape = (1088, 1920, 3)

	for raw in os.listdir(no_critter):
		# load image pixels
		image = cv2.resize(imread(no_critter + raw), (120, 68))
		print(image.shape)
		data.append(np.array(image))
		# 0 for no critter 
		labels.append(np.array([1, 0]))
		# image.shape = (1088, 1920, 3)
	data = np.array(data)
	labels = np.array(labels)
	return data, labels

data, labels = load_data()

# (2308,)
print(data.shape) 
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=101)

print(X_train.shape) # (1846,)
print(X_test.shape)
print(y_train.shape) # (462,)
print(y_test.shape)

# Plot 9 images
for i, image in enumerate(X_train[:9]):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(image)
	print('image', image.shape, 'label', y_train[i])
# show the figure
pyplot.show()
 

dropout = 0.2
model = Sequential()
# Reshape image to a much smaller size

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(68, 120, 3)))
model.add(Activation('relu'))

'''
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))
'''
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(2))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
				optimizer=opt,
				metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=3, epochs=50) # Causes error

pred = model.predict(X_test)

print('Test accuracy', accuracy_score(y_test, pred)*100)
