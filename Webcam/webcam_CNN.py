# Model for running with webcam; removes unecessary code from the full model

import numpy as np
import h5py
from APL import APLUnit
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, InputLayer
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

seed = 7
np.random.seed(seed)

#def frac_max_pool(x):
#    return tf.nn.fractional_max_pool(x,p_ratio)[0]

def model(weights = None, S = 5, p_ratio = [1.0, 2.6, 2.6, 1.0]):

    model = Sequential()

    model.add(Dropout(0.0, input_shape=(48, 48, 1)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (5, 5)))
    model.add(APLUnit(S=S))
    model.add(MaxPooling2D((3,3), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (5, 5)))
    model.add(APLUnit(S=S))
    model.add(InputLayer(input_tensor = tf.nn.fractional_max_pool(model.layers[7].output, p_ratio, overlapping=True)[0]))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (4, 4)))
    model.add(APLUnit(S=S))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(4096))
    model.add(APLUnit(S=S))
    model.add(Dense(6, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    if weights:
        model.load_weights(weights)

    return model

# build the model
model = model('../VGG16_regular_ninth_try2_PRIVATE_TEST.h5') # my weights

test_datagen = ImageDataGenerator(rescale = 1./255)

def prediction(img):
    prediction_generator = test_datagen.flow(img, [1])

    # uncomment for testing with actual image instead of webcam
    """
    prediction_generator = test_datagen.flow_from_directory(
            img,
            target_size = (48, 48),
            color_mode = 'grayscale')
    """
    return model.predict_generator(prediction_generator, 1)

# For when I want to test using an actual image in /images
"""
pred_array = prediction('./images')[0]

print(pred_array)
print("Angry: ", pred_array[0], "\nFear: ", pred_array[1], "\nHappy: ", pred_array[2], "\nSad: ", pred_array[3], "\nSurprise: ", pred_array[4], "\nNeutral: ", pred_array[5])
"""
