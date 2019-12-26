import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import keras

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
# (x_train, _), (x_test, y_test) = mnist.load_data()

data_int = np.random.random((900, 384))
data_float = np.random.random((900, 400))

# data pre-processing
# x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
# x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
# x_train = x_train.reshape((x_train.shape[0], -1))
# x_test = x_test.reshape((x_test.shape[0], -1))
# print(x_train.shape)
# print(x_test.shape)

# in order to plot in a 2D figure
encoding_dim = 20

# this is our input placeholder
# input_img = Input(shape=(784,))
input_int = Input(shape=(384,))
input_float = Input(shape=(400,))
input_layer = keras.layers.concatenate([input_int, input_float])

# encoder layers
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# decoder layers
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

# construct the autoencoder model
autoencoder = Model(input=[input_int,input_float], output=decoded)

# construct the encoder model for plotting
encoder = Model(input=[input_int,input_float], output=encoder_output)

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit([data_int,data_float], np.append(data_int,data_float,axis=1),
                nb_epoch=5,
                batch_size=256,
                shuffle=True)

# plotting
# encoded_imgs = encoder.predict(x_test)
# plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
# plt.colorbar()
# plt.show()


test = encoder.predict([data_int,data_float])
print(test.shape)
