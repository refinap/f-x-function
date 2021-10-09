"""
Refin Ananda Putra
github.com/refinap

"""

#create network 
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Dense
from tensorflow.keras.optimizers import SGD

#generate data from -20, -19.5, ..., 20
#ambil data dari -20  sampe 20 dengan beda 0.5
train_x = np.arange(-20, 20, 0.25)
#hitung target : sqrt(2x^2 +1)
train_y = np.sqrt((2*train_x**2)+1)

#Architecture taht will use
inputs = Input(shape=(1,)) #1 input Node
h_layer = Dense(8, activation='relu')(inputs) #8 node ar Hidden layer 1 with ReLU activation
h_layer = Dense(4, activation='relu')(h_layer) #4 node ar Hidden layer 2 with ReLU activation
output = Dense(1, activation='linear')(h_layer) # i output node with Linear activation
model = Model(inputs=inputs, outputs=output)

#update Rule or optimizer (loss function)
sgd = SGD(lr=0.001)
#compile the model with mean squared eror loss
model.compile(optimizer=sgd, loss='mse')

#Model is already, we can training the data use fit methode
#train the network and save the weights after training
model.fit(train_x, train_y, batch_size=20, epochs=1000, verbose=1) #batch_size 20 (mibi-batch SGD), do 1000 epoch and 
model.save_weights('weights.h5') #save all parameter (weight and bias) to file

#Training data prediction
#Prediction for number outside Training Data, 26, and will compare Training Data Prediction Result with Target
predict = model.predict(np.array([26]))
print('f(26) =' , predict)

predict_y = model.predict(train_x)

#Draw taget v prediction
plt.plot(train_x, train_y, 'r')
plt.plot(train_x, predict_y, 'b')
plt.show()
