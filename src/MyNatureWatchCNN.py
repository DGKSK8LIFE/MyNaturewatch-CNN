from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
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
		if np.all(image.shape == (68, 120, 3)):
			data.append(np.array(image))
			# 1 for yes critter
			labels.append(np.array([0, 1]))
		# image.shape = (1088, 1920, 3)

	for raw in os.listdir(no_critter):
		# load image pixels
		image = cv2.resize(imread(no_critter + raw), (120, 68))
		if np.all(image.shape == (68, 120, 3)):
			data.append(np.array(image))
			# 0 for no critter 
			labels.append(np.array([1, 0]))
			# image.shape = (1088, 1920, 3)
	data = np.array(data)
	labels = np.array(labels)
	return data, labels

data, labels = load_data()

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=42)

model = Sequential()
# Reshape image to a much smaller size

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(68, 120, 3)))
model.add(Activation('relu'))


model.add(Conv2D(32, (3, 3)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = RMSprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
				optimizer=opt,
				metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=3, epochs=50)

pred = model.predict(X_test).round()

print('Test accuracy', accuracy_score(y_test, pred)*100)

answer = None 
while answer not in ['yes', 'y', 'no', 'n']:
	answer = input("Would you like to save this model (and overwrite the previously saved one)? (y/n)")
	if answer == 'yes' or answer == 'y':
		model.save('model/MyNaturewatchCNN')
		break
	elif answer == 'no' or answer == 'n':
		break
	else:
		print('Please enter yes or no')

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
print('False positive - non-animal photo that would be saved', false_positives)
print('False negative - animal photo that would get deleted', false_negatives)

