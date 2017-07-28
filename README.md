# Emotion Predicting CNN
This was my capstone project for Udacity's Machine Learning Nanodegree.

## Emotion Classification with Deep Learning
In this project, I used various deep learning techniques that are well known like convolutional, padding, and pooling layers. However, I also incorporate [Fractional Max Pooling (FMP)](https://arxiv.org/abs/1412.6071) layers and program [Adaptive Piecewise Linear (APL)](https://arxiv.org/pdf/1412.6830.pdf) units/activation functions in Keras.

For my full report, you can read it [here](https://github.com/bartchr808/machine-learning/blob/master/projects/capstone/Submission/Report.pdf).

For my capstone proposal, you can read it [here](https://github.com/bartchr808/machine-learning/blob/master/projects/capstone/Submission/Proposal.pdf)
## Dataset

The dataset can be attained [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). It is a .csv file where the image is stored as a list of numbers in a single cell, but the script *csv2image.py* can convert them to 35,887 48x48-pixel grayscale images of faces. Of these images, there are 7 classes:
* Angry (4953 entries)
* Disgust (547 entries)
* Fear (5121 entries)
* Happy (8989 entries)
* Sad (6077 entries)
* Surprise (4002 entries)
* Neutral (6198 entires)

As you can see, there are very few examples of Disgust, thus my model has only been trained on the other 6 labels.
## Setting up Environment

You will need the following libraries to get everything running:
* OpenCV/cv2
* Matplotlib
* PIL
* h5py
* Python 3.x
* Keras
* Tensorflow
* Numpy

If running into issue with not being able to `import name weakref`, run: `pip install backports.weakref`.

If getting error `No module named google.protobuf`, run: `conda install protobuf`.

If there are any problems you are having, please submit an issue.

## Key Files Descriptions
* **APL.py:** my implementation of the APL activation unit.
* **batch_fscore.py:** my implementation of F-score.
* **csv2image.py:** script to convert the .csv file from Kaggle to saved images. It also saves it to the correct folders and correct folder structure, depending on if it's a part of the Training, PrivateTest, or PublicTest dataset.
* **good_webcam.mp4:** video of the webcam working.
* **model.py** my CNN I trained.
* **Proposal.pdf:** my proposal submission.
* **Report.pdf:** my report.
* **webcam_capture.py** script that captures images, gets model's predictions, and display them in a Pyplot.
* **webcam_CNN.py:** same as model.py but some things removed and implemented to work with webcam_capture.py like new prediction method.
* **webcam.gif:** same as good_webcam.mp4 but in gif format

# Training the model on your own data
If you aren't familiar with Keras' function `flow_from_directory`, put your training set into a folder called *Training* and your testing set into a folder called *PublicTest*. Instead, try running *csv2image.py* on the dataset *fer2013* as linked above and see how the folders are structured. For example:

>Training
>>0
>>1
>>2
>>3
>>4
>>5

where

* 0 -> Angry
* 1 -> Fear
* 2 ->Happy
* 3 -> Sad
* 4 -> Surprise
* 5 -> Neutral


If you want to save your weights, modify line 97 in *model.py`:
```
save_best = ModelCheckpoint('name_of_your_weights.h5', monitor='val_acc', verbose=2, save_best_only=True, mode='max')
```
and it will save the weights with the highest validation accuracy over your epochs.

## Running the Webcam CNN
