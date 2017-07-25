# All necessary imports
import numpy as np
import h5py
from APL import APLUnit
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, InputLayer
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from batch_fscore import fbeta_score
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

seed = 7
np.random.seed(seed)
training_size = 28273
validation_size = 3533 # size of either test set

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

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', fbeta_score])

    if weights:
        model.load_weights(weights)

    return model


# build the model
model = model('VGG16_regular_ninth_try2_PRIVATE_TEST.h5') # my weights

batch_size = 256

# Preprocess inputted data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range = 30,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Fit the model
train_generator = train_datagen.flow_from_directory(
        '../Training',  # this is the target directory
        target_size = (48, 48),  # all images will be resized to 48x48
        batch_size = batch_size,
        color_mode = 'grayscale')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '../PrivateTest',
        target_size = (48, 48),
        batch_size = batch_size,
        color_mode = 'grayscale')


# ~~~~~~~~~~~~~~~~ Check accuracy & F-score ~~~~~~~~~~~~~~~
score = model.evaluate_generator(validation_generator, validation_size // batch_size)
print("TEST")
print(score)
print("Loss: {0:.3} \nAccuracy: {1:.3%} \nF-Score: {2:.3%}").format(score[0], score[1], score[2])


# ~~~~~~~~~~~~~~~~~~~~~~ Train model ~~~~~~~~~~~~~~~~~~~~~~
# callback functions
save_best = ModelCheckpoint('VGG16_regular_ninth_try2_PRIVATE_TEST2.h5', monitor='val_acc', verbose=2, save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
              patience=5, min_lr=0.001, verbose=1)

model.fit_generator(
        train_generator,
        steps_per_epoch = training_size // batch_size,
        epochs=30,
        callbacks = [save_best, reduce_lr],
        validation_data=validation_generator,
        validation_steps= validation_size // batch_size)
#model.save_weights('VGG16_regular_second_try.h5')  # always save your weights after training or during training
