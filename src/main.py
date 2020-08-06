from tensorflow.keras.models import load_model
from paramiko import SSHClient
from scp import SCPClient
from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import cv2

model = load_model('model/MyNaturewatchCNN')

temp = '/tmp/photos'
inputs = []

for raw in os.listdir(temp):
    image = cv2.resize(imread(temp + raw), (120, 68))
    if np.all(image.shape == (68, 120, 3)):
        inputs.append(np.array(image))

