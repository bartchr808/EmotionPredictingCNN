# coding: utf-8

# incase python can't find cv2
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import os
import cv2
import numpy as np
from PIL import Image
from webcam_CNN import prediction
from matplotlib import pyplot as plt

vidcap = cv2.VideoCapture(0)

def crop_center(img, cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

while (True):
    retval, image = vidcap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crop image
    width, height = np.array(image).shape
    new_width, new_height = (350, 350) # 1280 × 720 before
    cropped = crop_center(image, new_width, new_height)

    # Predictions
    input_data = np.resize(cropped, [1, 48, 48, 1])
    pred_array = prediction(input_data)[0]

    print "Angry: ", pred_array[0], "\nFear: ", pred_array[1], "\nHappy: ", pred_array[2], "\nSad: ", pred_array[3], "\nSurprise: ", pred_array[4], "\nNeutral: ", pred_array[5]


    # Plot
    fig, ax1 = plt.subplots()
    left, bottom, width, height = [0.225, 0.73, 0.15, 0.15]
    ax2 = fig.add_axes([left, bottom, width, height])

    plt.tick_params(
    axis='both',
    which='both',
    labelleft='on',
    labelbottom='off',
    top='off',
    bottom='off',
    left='off')
    ax1.tick_params(
    axis='both',
    which='both',
    labelleft='off',
    labelbottom='off',
    top='off',
    bottom='off',
    left='off')

    ax1.imshow(cropped, cmap="gray", vmin=0, vmax=255)

    Emotions = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    y_pos = np.arange(len(Emotions))
    ax2.barh(y_pos, pred_array, align='center', color='g')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(Emotions)
    ax2.patch.set_alpha(0.) # transparent
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    plt.pause(0.0001) # random tiny timeout value to start loop over again
