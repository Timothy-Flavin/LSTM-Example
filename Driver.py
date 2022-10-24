# General libraries to plot stuff and work with data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Methods made in this directory to format failure data
from DataLoader import LoadAsDataFrame, CreateSequentialData, NormalizeData, ReshapeData, TransferSplit

# These are the files I had preped for this example. Feel 
# free to add more of this format. These parameters change
# how the model is trained so for multiple runs you can 
# change these or set them equal to either input() or cmd
# line arguments for testing purposes. This code is not the
# whole project, just a playground for transfer learning that
# you can run and edit as you like. Good luck

seqLength = 5 # How many failures the LSTM gets to see before making a prediction
colname = "cum_f_time" # which pattern to learn, cumulative or inter-failure time
validationSet = 3 # Which dataset to use to test transfer learning, 3 => SS1Data
trainProp = 0.5 # What percent of the transfer learning dataset to use for tuning
DataNames = [ #W hich files to use
  "CSR1Data",
  "CSR2Data",
  "CSR3Data",
  "SS1Data",
  "SS3Data",
]



rawData = []

# First we load all the failure data into some dataframes
for name in DataNames:
  rawData.append(LoadAsDataFrame(name, verbose=False))

# Next we will create a dataframe that is in the format that 
# Recurrent networks want. This format is explained in a 
# comment above DataLoader.CreateSequentialData

seqData = []
for d in rawData:
  seqData.append(CreateSequentialData(d,colname, seqLength, False))

# Part of the challenge of transfer learning is that we want the
# data to be very similar so that our model can understand it, but
# Some of the failure times data is in miliseconds, some seconds,
# and some other amounts of time and to add to that the number of
# samples is different for each so to the model the scale will be
# all over the place. We want to normalize the data in some way
# so that the model can generalize, but we can't use any of the 
# information from test data that is held-out from the model. For
# Time series data we want the first X % to be training data and 
# The last (100-X)% to be testing data. What we can do is take the
# cumulative failure times and divide them all by the time that is
# X % down from the top of the table. So the training data will 
# always climb from 0 to 1, and the testing data will climb from
# 1 up. In addition, we can divide the f_num by the X % f_num 
# in order to plot them all with the same axis. 

normalizedData = []
for i,s in enumerate(seqData): #enumerate lets you get both index and
  normalizedData.append(NormalizeData(s,trainProp, verbose=False))

# This is a sanity check to make sure it looks right
print("Raw data [0] first few rows: ")
print(rawData[0].head())
print("Raw data After being turned into sequences: ")
tempDat = CreateSequentialData(rawData[0],"cum_f_time", seqLength, True)
print("Raw data After being normalized: ")
NormalizeData(tempDat,trainProp, verbose=True)


# Below is the graph of all data before and after normalization:
# Before:
for d in rawData:
  plt.plot(d["cum_f_time"])
  plt.legend(DataNames)
plt.title("Raw data cumulative failure time")
plt.grid()
plt.show()

# After:
for n in normalizedData:
  plt.plot(n["f_num"],n["cum_f_time"])
  plt.legend(DataNames)
plt.title("Normalized data cumulative failure time")
plt.grid()
plt.show()


# Re doing normalization to only scale the validation dataset to be
# between 0 and 1 for train prop. All the other datasets will be 
# considered training data for the transfer learner so they will
# have a train prop of 1.0 instead of trainProp
normalizedData = []
for i,s in enumerate(seqData): #enumerate lets you get both index and
  if(i==validationSet):
    normalizedData.append(NormalizeData(s,trainProp, verbose=False))
  else:
    normalizedData.append(NormalizeData(s,1.0, verbose=False))


# Because we only have one input feature, our table row looks like this
# x: [1,2,3] y: 4
# but with multiple features like x1 and x2 coordinates it would have to
# look like this: x: [[1,4][2,3][3,2][4,1]], y[5,0]. Because LSTM needs
# to be able to do both, we need to re format out data to be
# x:[[1],[2],[3]] y:[4] so this function will return numpy arrays with
# the proper x and y dimensions and structure. Particularly, this func
# will return a two variables, x and y where x is np array of sequences
# numpy can do this with the "reshape" or "expand_dims" function
reshapedData = []
reshapedTargets = []
for n in normalizedData:
  x, y = ReshapeData(n, colname, False)
  reshapedData.append(x)
  reshapedTargets.append(y)


# Last bit of data processing is to split the data into train
# and test sets so that the models can learn on some data and 
# then be validated on other data. For transfer learning we 
# want to allow a model to see one or more sample datasets as
# training data and then we want to try it on a new dataset
# or tune it to that dataset a little and see how it goes. 
# The article I sent you explains this in more depth, but the
# main advantages possible from transfer learning are that a
# model with previous expirience usually does better when data
# is limited, so if you give very few datapoints of the new 
# failure data to the model, it should be able to adjust it's 
# weights a little and do well instead of training a lot from
# random weights. The other benifit is a pre-trained model may
# need no tuning at all, or it may converge very quickly so we
# want to measure accuracy on small amounts of training data
# as well as training time compared to a model that has not 
# been tuned at all. Automating that comparison will be left
# to the reader ;) but I made a function with some tunable
# parameters so you can run this a few times with different 
# settings as a good start. If you have time to take it further
# and do a grid search or filter search of the parameters that
# would be cool. Otherwise you may want to try switching which 
# dataset is used as the tester to compare the transfer learner
# with the default LSTM. 

# Col: collective data for transfer learner
# Tun: first portion of validation dataset for both transfer
#      learner and normal learner or just normal learner
# Test: test for performance to be used on both learners  
xCol, yCol, xTun, yTun, xTest, yTest = TransferSplit(reshapedData, reshapedTargets, 3, trainProp)

# Time to define our model
from MyKerasModel import getModel

print("Getting models")
model1 = getModel(100,seqLength=seqLength)
model2 = getModel(100,seqLength=seqLength)
model3 = getModel(100,seqLength=seqLength)

print("Training models")
# The commented out line uses a validation set to track the model progress.  

# col1hist = model1.fit(xCol[:800], yCol[:800], epochs=50, validation_data=(xCol[800:], yCol[800]))
col1hist = model1.fit(xCol, yCol, epochs=30, validation_data=(xTest, yTest))#)
col2hist = model2.fit(xCol, yCol, epochs=30, validation_data=(xTest, yTest))#, verbose=0)
tun1hist = model2.fit(xTun, yTun, epochs=30, validation_data=(xTest, yTest))#, verbose=0)
tun2hist = model3.fit(xTun, yTun, epochs=30, validation_data=(xTest, yTest))#, verbose=0)

# Now that model 1 and 2 have been trained on collective data and 
# model 2 and 3 have been tuned, time to graph the results
# First lets graph the history of training loss to see how fast 
# models converged. For your project use validation loss similar to
# the commented out line 164

print(col1hist.history['loss'])

plt.title("Training Loss over time")
plt.plot(col1hist.history['loss'])
plt.plot(col2hist.history['loss'])
plt.plot(tun1hist.history['loss'])
plt.plot(tun2hist.history['loss'])
plt.legend(["Model 1 Col Loss","Model 2 Col Loss","Model 2 Tun Loss","Model 3 Tun Loss"])
plt.grid()
plt.show()

plt.title("Test Data Loss over time")
plt.plot(col1hist.history['val_loss'])
plt.plot(col2hist.history['val_loss'])
plt.plot(tun1hist.history['val_loss'])
plt.plot(tun2hist.history['val_loss'])
plt.legend(["Model 1 Col Loss","Model 2 Col Loss","Model 2 Tun Loss","Model 3 Tun Loss"])
plt.grid()
plt.show()

plt.plot(np.reshape(yTest,(yTest.shape[0],)))
plt.plot(np.reshape(model1.predict(xTest), (yTest.shape[0],)))
plt.plot(np.reshape(model2.predict(xTest), (yTest.shape[0],)))
plt.plot(np.reshape(model3.predict(xTest), (yTest.shape[0],)))
plt.grid()
plt.legend(["Actual","Untuned","Tuned","No Transfer"])
plt.show()
print()