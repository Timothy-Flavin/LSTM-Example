import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Get the data as a pandas dataframe with three rows
# failure_number, interfailure_time, cumulative_interfailure_time
# Cumulative failure time is usefull because it is a little less
# Noisy looking and it looks like a curve that levels off 
# instead of a spikey mess
def LoadAsDataFrame(fileName, verbose=False):
  data = pd.read_csv("Data/"+fileName+".csv", names=["f_num","f_time"])
  data["cum_f_time"] = np.cumsum(data["f_time"])
  data = data.astype("float32")
  
  if verbose:
    print(data.head())
  
  return data


# We have to give the lstm sequences of data to look at so that it
# can take advantage of it's recurrent power. What this means is
# we give it a list of values over time and it predicts the next
# value so like if our data was [[1],[2],[3],[4],[5],[6]] we would 
# want to give the lstm [[1],[2],[3]] and it would be fed the number [1]
# then it would calculate a hidden state, then it would see [2] then
# update it's hidden state, then see [3] and update again and make it's
# prediction which we would hope to be [4]. The sequence length here
# is how many falure times it get's to see before making a prediction
# so if seqLength is 3, then we want to create a dataset like this:
#   X_data      Y_data/target
# | [1][2][3] | [4] |
# | [2][3][4] | [5] |
# | [3][4][5] | [6] |
#
# A longer sequence means fewer datapoints total but more information
# for the model to work with and train on. If we want the model to make
# multiple predictions out into the future we just feed it it's own
# predictions so say we give it 1,2,3 and want it to predict 4,5,6
#   X_data               model_prediction
# | [1][2][3]           | [4.1] |
# | [1][2][3][4.1]      | [4.9] |
# | [3][4][5][4.1][4.9] | [6.0] |
# This function puts the data into the form above so we can train
# The model. colname can either be "f_time" or "cum_f_time" 
def CreateSequentialData(df, colname, seqLength, verbose=False):
  temp_dat = df[["f_num",colname]]
  for i in range(seqLength-1,-1,-1):
    temp_dat[seqLength-i-1]=temp_dat[colname].shift(i+1)
  temp_dat = temp_dat.iloc[seqLength:]

  if verbose:
    print(temp_dat.head())

  return temp_dat


# TrainProp is the proportion of the data to be
# Used for training. It will be between 0 and 1
def NormalizeData(df, trainProp, colname="cum_f_time", verbose=False):
  newData = df.copy()
  index = int(trainProp*len(newData.index))
  newData["f_num"] /= newData["f_num"][index]
  cumTime = newData[colname][index]
  for i in newData.columns:
    if i=="f_num":
      continue
    else:
      newData[i] = newData[i]/cumTime
  
  if verbose:
    print(newData.head())

  return newData


# Turns the dataframe into 2 numpy arrays that are the
# proper dimensions for an LSTM to train on. 
def ReshapeData(df, colname, verbose=False):
  if verbose:
    print("Reshaping data")
    print(df.head())
  data = df.copy().drop(columns=["f_num"])
  if verbose:
    print(data.head())

  # x is all the columns but the target column and 
  x = np.expand_dims(data.drop(columns=[colname]).to_numpy(), axis=-1)
  y = np.expand_dims(data[colname].to_numpy(), axis=-1)

  if verbose:
    print(x)
    print(y)
    input("Press key to continue")

  return x,y


# Splits the data into 3 groups, collective, tune and test. xCol, xTun, xTest, 
# yCol, yTun, and yTest. Col and Tun will be used on the transfer learner,
# only Tun will be used on the normal LSTM, and test will be used to validate
# both
def TransferSplit(xDataSets, yDataSets, vnum, trainProp, verbose=False):
  # Making some empty numpy arrays to hold all the data we want
  # xshape = (numExamples, seqLength, numFeatures) => (0,seqLength,1)
  xCol = np.empty(dtype='float32',shape=(0, xDataSets[0].shape[1], 1))
  xTun = np.empty(dtype='float32',shape=(0, xDataSets[0].shape[1], 1))
  xTest = np.empty(dtype='float32',shape=(0, xDataSets[0].shape[1], 1))

  yCol = np.empty(dtype='float32',shape=(0, 1))
  yTun = np.empty(dtype='float32',shape=(0, 1))
  yTest = np.empty(dtype='float32',shape=(0, 1))

  for i in range(len(xDataSets)):
    # Validation set
    print(f"X dataset {i} shape: {xDataSets[i].shape}")
    print(f"Y dataset {i} shape: {yDataSets[i].shape}")
    if i==vnum:
      xTun = xDataSets[i][:int(trainProp*xDataSets[i].shape[0]),:,:]
      xTest = xDataSets[i][int(trainProp*xDataSets[i].shape[0]):,:,:]
      yTun = yDataSets[i][:int(trainProp*yDataSets[i].shape[0]),:]
      yTest = yDataSets[i][int(trainProp*yDataSets[i].shape[0]):,:]
    else:
      xCol = np.append(xCol, xDataSets[i], axis=0)
      yCol = np.append(yCol, yDataSets[i], axis=0)
  print(f"xCol shape: {xCol.shape}")
  print(f"yCol shape: {yCol.shape}")
  print(f"xTun shape: {xTun.shape}")
  print(f"yTun shape: {yTun.shape}")
  print(f"xTest shape: {xTest.shape}")
  print(f"yTest shape: {yTest.shape}")

  return xCol, yCol, xTun, yTun, xTest, yTest