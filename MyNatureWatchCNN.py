from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import os
import numpy as np
import cv2
import pandas as pd

base = '/home/jose/Programming/aiml/Data/naturewatch'
# Directory of all the pictures with an animal 
critter = base + '/critter/'
# Directory of all the pictures without an animal
no_critter = base + '/no_critter/'


def plot_image(index):
	# Plot 9 images
	for i, image in enumerate(X_train[index]):
		plt.imshow(image)
		print('image', image.shape, 'label', y_train[i])
	# show the figure
	plt.show()
	

def load_data():
	data = []
	labels = []
	for raw in os.listdir(critter):
		# The array of values
		image = cv2.resize(imread(critter + raw), (120, 68))
		data.append(np.array(image))
		# 1 for yes critter
		labels.append(np.array([0, 1]))
		# image.shape = (1088, 1920, 3)

	for raw in os.listdir(no_critter):
		# load image pixels
		image = cv2.resize(imread(no_critter + raw), (120, 68))
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


dropout = 0.2
model = Sequential()
# Reshape image to a much smaller size

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(68, 120, 3)))
model.add(Activation('relu'))


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


model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=3, epochs=20)

pred = model.predict(X_test).round()

print('Test accuracy', accuracy_score(y_test, pred)*100)

def plot_one_image(idx):
	print('label: %s' % ('yes animal' if np.all(y_test[idx] == np.array([0,1])) else 'no animal'))
	print('guessed: %s' % ('yes animal' if np.all(pred[idx] == np.array([0,1])) else 'no animal'))
	print()
	plt.imshow(X_test[idx])
	plt.show()
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0


for i in range(len(y_test)):
	# True positive. Label: critter, prediction: critter.
	if np.all(pred[i] == y_test[i]) and np.all(pred[i] == np.array([0, 1])):
		true_positives += 1
		
	# True negative. Label: no critter, no prediction: no critter.
	elif np.all(pred[i] == y_test[i]) and np.all(pred[i] == np.array([1, 0])):
		true_negatives += 1
	
	# False positive. Label: no critter, prediction: critter.
	elif np.all(pred[i] != y_test[i]) and np.all(pred[i] == np.array([0, 1])):
		print('False positive:')
		plot_one_image(i)
		false_positives += 1
	
	# False negative. Labels: critter, prediction: no critter.
	elif np.all(pred[i] != y_test[i]) and np.all(pred[i] == np.array([1, 0])):
		print('False negative:')
		plot_one_image(i)
		false_negatives += 1

print("True postitive", true_positives)
print('True negatives', true_negatives)
print('False positive', false_positives)
print('False negative', false_negatives)
