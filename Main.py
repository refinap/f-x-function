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

#ambil data dari -20  sampe 20 dengan beda 0.5
train_x = np.arange(-20, 20, 0.25)
#hitung target : sqrt(2x^2 +1)
train_y = np.sqrt((2*train_x**2)+1)

inputs = Input(shape=(1,))
h_layer = Dense(8, activation='relu')(inputs)
h_layer = Dense(4, activation='relu')(h_layer)
output = Dense(1, activation='linear')(h_layer)
model = Model(inputs=inputs, outputs=output)

#update Rule
sgd = SGD(lr=0.001)
#compile the model with mean squared eror loss
model.compile(optimizer=sgd, loss='mse')

#train the network and save the weights after training
model.fit(train_x, train_y, batch_size=20, epochs=1000, verbose=1)
model.save_weights('weights.h5')

#prediksi training data
predict = model.predict(np.array([26]))
print('f(26) =' , predict)

predict_y = model.predict(train_x)

#gambar taget v prediction
plt.plot(train_x, train_y, 'r')
plt.plot(train_x, predict_y, 'b')
plt.show()
