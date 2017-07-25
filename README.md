# EmotionPredictingCNN

## Dataset

The dataset can be attained here: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## Setting up Environment

Needs Opencv, Matplotlib, PIL, h5py, python 2.7, Keras, Tensorflow, and some other standard libraries like Numpy.

If running into issue with not being able to import name weakref, run: `pip install backports.weakref`.

If getting error `No module named google.protobuf`, run: `conda install protobuf`.

## File Description
* *APL.py:* my implementation of the APL activation unit.
* *batch_fscore.py:* my implementation of F-score.
* *csv2image.py:* script to convert the .csv file from Kaggle to saved images.
* *good_webcam.mp4:* video of the webcam working.
* *model.py* my CNN I trained.
* *Proposal.pdf:* my proposal submission.
* *Report.pdf:* my report.
* *webcam_capture.py* script that captures images, gets model's predicitons, and display them in a Pyplot.
* *webcam_CNN.py:* same as model.py but some things removed and implemented to work with webcam_capture.py like new prediction method.
* *webcam.gif:* same as good_webcam.mp4 but in gif format
