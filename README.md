# EmotionPredictingCNN
This was my capstone project for Udacity's Machine Learning Nanodegree.

The model makes the following predictions on the image below:
* Angry:  0.00982473%
* Fear:  0.206567%
* **Happy:  97.8469%**
* Sad:  0.934201%
* Surprise:  0.00566902%
* Neutral:  0.0242221%

![Happy Image](https://github.com/bartchr808/EmotionPredictingCNN/blob/master/Media/happy.png "Happy image")

## Emotion Classification with Deep Learning
In this project, I used various deep learning techniques that are well known like convolutional, padding, and pooling layers. However, I also incorporate [Fractional Max Pooling (FMP)](https://arxiv.org/abs/1412.6071) layers and programmed [Adaptive Piecewise Linear (APL)](https://arxiv.org/pdf/1412.6830.pdf) units/activation functions in Keras.

For my full report, you can read it [here](https://github.com/bartchr808/machine-learning/blob/master/projects/capstone/Submission/Report.pdf).

For my capstone proposal, you can read it [here](https://github.com/bartchr808/machine-learning/blob/master/projects/capstone/Submission/Proposal.pdf)

Overall, it unoffically ranked **5th** on the Kaggle leaderboards for [this competition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/leaderboard) with an F-score of **65.20%** and accuracy of **65.39%**.

Generally this would be considered a terrible model, but relative to the Kaggle leaderboards, this is a great model. This is also apparent when you take into account some of the difficulties in the dataset (read my full report for more detail) and how hard it is to read emotions on people since many people express them differently. Finally, some emotions are sometimes very similar to people's other ones and only very minute differences separate them.


## Setting up Environment
You will need the following libraries to get everything running (to avoid running into issues, try running the following in a Conda environment by downloading [Miniconda](https://conda.io/miniconda.html)):
* OpenCV/cv2
* Matplotlib
* PIL
* h5py
* Python 3.x
* Keras
* Tensorflow
* Numpy

If running into issue with not being able to `import name weakref`, run: `pip install backports.weakref`. If getting error `No module named google.protobuf`, run: `conda install protobuf`.

Once all environment has been setup, run the following:
```
git clone https://github.com/bartchr808/EmotionPredictingCNN
cd EmotionPredictingCNN
```
If there are any problems you are having, please submit an issue.

## Key Files Descriptions
* **APL.py:** my implementation of the APL activation unit.
* **batch_fscore.py:** my implementation of F-score.
* **csv2image.py:** script to convert the .csv file from Kaggle Fer2013 dataset to saved images. It also saves it to the correct folders and correct folder structure, depending on if it's a part of the Training, PrivateTest, or PublicTest set.
* **good_webcam.mp4:** video of the webcam working.
* **model.py** my CNN I trained.
* **webcam_capture.py** script that captures images, gets model's predictions, and display them in a Pyplot.
* **webcam_CNN.py:** same as model.py but some things removed and implemented to work with webcam_capture.py like new prediction method.
* **webcam.gif:** same as good_webcam.mp4 but in gif format.

## Dataset
The dataset can be attained [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). It is a .csv file where the image is stored as a list of numbers in a single cell, but the script *csv2image.py* can convert them to 35,887 48x48-pixel grayscale images of faces. Of these images, there are 7 classes:
* Angry (4953 entries)
* Disgust (547 entries)
* Fear (5121 entries)
* Happy (8989 entries)
* Sad (6077 entries)
* Surprise (4002 entries)
* Neutral (6198 entires)

As you can see, there are very few examples of Disgust, thus my model has only been trained on the other 6 labels. Thus, run:
```
python csv2image.py
```
And it will create a folder called _Training_, _PrivateTest_, and _PublicTest_ that will house all the images.

## Training the model on your own data
If you aren't familiar with Keras' function `flow_from_directory`, put your training set into a folder called *Training* and your testing set into a folder called *PublicTest*. Instead, try running *csv2image.py* on the dataset *fer2013* as linked above and see how the folders are structured. For example:
```
Training/
    0/
    1/
    2/
    3/
    4/
    5/
```
where:
* 0 -> Angry
* 1 -> Fear
* 2 ->Happy
* 3 -> Sad
* 4 -> Surprise
* 5 -> Neutral

If you want to save your weights, modify line 97 in *model.py`:
```
save_best = ModelCheckpoint('name_of_your_weights.h5',
monitor='val_acc', verbose=2, save_best_only=True, mode='max')
```
and it will save the weights with the highest validation accuracy over your epochs. Then, simply run:
```
python Model/model.py
```

## Running the Webcam CNN
Currently the program uses new Pyplots to display the images and predictions.

Run:
```
python Webcam/webcam_capture.py
```
And press `Ctrl+C` to interrupt and stop running the program.

![Old Emotion CNN gif](https://github.com/bartchr808/EmotionPredictingCNN/blob/master/Media/webcam.gif?raw=true "Old Emotion CNN gif")

## Known Bugs
* **Slow Predictions:** Currently, the Webcam CNN predictions are delayed by about 3 seconds relative to the displayed image. I'm currently working on a fix to not display the image in the Pyplot until the predictions have been computed. If you have a powerful GPU this shouldn't be an issue.
