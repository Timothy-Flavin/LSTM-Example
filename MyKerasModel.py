import tensorflow as tf
# define model
def getModel(units=50, seqLength=5):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.LSTM(units, activation='relu', input_shape=(seqLength, 1)))
  model.add(tf.keras.layers.Dense(1))
  model.compile(optimizer='sgd', loss='mae') #opt='adam' and loss='mse' optimizer='sgd', loss='mae'
  return model
