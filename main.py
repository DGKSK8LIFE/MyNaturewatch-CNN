from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot
from matplotlib.image import imread
import tensorflow as tf
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
		data.append(image)
		labels.append(1)

	for raw in os.listdir(no_critter):
		# load image pixels
		image = imread(no_critter + raw)
		data.append(image)
		labels.append(0)
	#data = np.asarray(data)
	#labels = np.asarray(labels)
	return data, labels

data, labels = load_data()

#assert data.shape[0] == labels.shape[0]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=101)

X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

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
model.add(Reshape((272, 480, 3)))

model.add(Conv2D(32, (3, 3), padding='same'))
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
model.add(Dense(512))
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
'''
for i in [X_train, X_test, y_train, y_test]:
	print(i.shape)
	print(i.shape[0])
	print(type(i))
	print()
'''

model.fit(X_train, y_train)

'''
pred = model.predict(X_test)

print('Test accuracy', accuracy_score(y_test, pred)*100)
'''
