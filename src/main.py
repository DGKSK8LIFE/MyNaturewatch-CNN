from tensorflow.keras.models import load_model
from paramiko import SSHClient
from scp import SCPClient
from matplotlib import pyplot as plt
from matplotlib.image import imread
from socket import gaierror
from os.path import join
from time import sleep
import numpy as np
import cv2
import sys
import os

temp = '/tmp/photos'
pictures_dir = '/home/jose/Pictures/MyNaturewatch'
'''
def progress(filename, size, sent):
    sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent)/float(size)*100) )

try:
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname='camera.local', 
                username='pi',
                password='badgersandfoxes')

    scp = SCPClient(ssh.get_transport(), progress=progress)

    scp.get('~/NaturewatchCameraServer/www/photos', '/tmp', recursive=True)

    assert len(os.listdir(temp)) != 0

    stdin, stdout, stderr = ssh.exec_command('cd ~/NaturewatchCameraServer/www/photos && rm -f *')
    sleep(5)
    for line in stdout.readlines(): print(line)
    for line in stderr.readlines(): print(line)

    ssh.close()
except gaierror:
    print("Name or service not known: Are you connected to the camera's internet?")
'''
temp_list = [raw for raw in os.listdir(temp) if raw.endswith('.jpg')]
input_images = []

for raw in temp_list:
    image = cv2.resize(imread(join(temp, raw)), (120, 68))
    if np.all(image.shape == (68, 120, 3)):
        input_images.append(np.array(image))

input_images = np.array(input_images)

model = load_model('/home/jose/Programming/naturewatch-cnn/model/MyNaturewatchCNN')

preds = model.predict(input_images).round()

assert len(temp_list) == len(preds)

for i in range(len(preds)):
    date = (temp_list[i])[:10]

    if not os.path.exists(join(pictures_dir, date)):
        os.system('mkdir ' + join(pictures_dir, date))
    
    if not os.path.exists(join(pictures_dir, date) + '-no-critter'):
        os.system('mkdir ' + join(pictures_dir, date) + '-no-critter')

    # Critter
    if np.all(preds[i] == np.array([0, 1])):
        # Ex: mv /temp/photos/2020-04-05-14-15-23.jpg /home/jose/Pictures/MyNaturewatch/2020-04-05
        os.system('mv ' + join(temp, temp_list[i]) + ' ' + join(pictures_dir, date))
    #No critter
    else:
        os.system('mv ' + join(temp, temp_list[i]) + ' ' + join(pictures_dir, date) + '-no-critter')
