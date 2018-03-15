# NotHotDogCNN
NotHotDog using Convolutional Neural Network

Previously, I tried to get a Neural Network classify betweeen a hot dog and everything else. It was a simple neural network with 
just 1 hidden layer. 

In this project, I tried the same with a Convolutional Neural Network. It contains 2 convolutional layers trained against a set of 800 
examples.

It achieved about 79% accuracy which is pretty impressive considering it had only 800 images to train on. (On a simple neural network it achieved
about 55% accuracy.)
So this explains why Convolutional Neural Networks are used in image recognition.

P.S.
image_preprocessing.py file is the same as the previous one excpet this time it takes RGB color of every pixels to build the training data.
