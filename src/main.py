# ssh pi@camera.local
# pass: badgersandfoxes

# /NaturewatchCameraServer/www/photos
from tensorflow.keras.models import load_model
from paramiko import SSHClient
from scp import SCPClient
from matplotlib import pyplot as plt
from matplotlib.image import imread
from socket import gaierror
import numpy as np
import os
import cv2

temp = '/tmp/photos'

os.system('mkdir ' + temp)

try:
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname='camera.local', 
                #port = 'port',
                username='pi',
                password='badgersandfoxes')


    # SCPCLient takes a paramiko transport as its only argument
    scp = SCPClient(ssh.get_transport())

    scp.get('~/NaturewatchCameraServer/www/photos', temp, recursive=True)

    assert len(os.listdir(temp)) != 0

    scp.close()

    stdin, stdout, stderr = ssh.exec_command('rm -rf ~/NaturwatchCameraServer/www/photos/*.jpg')

    for line in stdout.readlines(): print(line)

    ssh.close()
except gaierror:
    print("Name or service not known: Are you sure you are connected to the camera's internet?")

inputs = []

for raw in os.listdir(temp):
    if raw.endswith('.jpg'):
        image = cv2.resize(imread(temp + raw), (120, 68))
        if np.all(image.shape == (68, 120, 3)):
            inputs.append(np.array(image))

#model = load_model('model/MyNaturewatchCNN')

