from tensorflow.keras.models import load_model
from paramiko import SSHClient
from scp import SCPClient
from matplotlib import pyplot as plt
from matplotlib.image import imread
from socket import gaierror
import numpy as np
import cv2
import sys
import os

temp = '/tmp/photos'

def progress(filename, size, sent):
    sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent)/float(size)*100) )

try:

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname='camera.local', 
                username='pi',
                password='badgersandfoxes')

    scp = SCPClient(ssh.get_transport(), progress=progress)

    scp.get('~/NaturewatchCameraServer/www/photos', temp, recursive=True)

    assert len(os.listdir(temp)) != 0

    stdin, stdout, stderr = ssh.exec_command('rm -rf ~/NaturwatchCameraServer/www/photos/*')

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
    else:
        os.system('rm ' + raw)

#model = load_model('model/MyNaturewatchCNN')

