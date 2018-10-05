import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils

#setting image data format
K.set_image_dim_ordering('th')

#loading data into variables
(X_train,y_train),(X_test,y_test) = mnist.load_data()



#reshaping and making images suitable for network
X_train = X_train.reshape([X_train.shape[0], 1, 28, 28]).astype('float32')/255
X_test = X_test.reshape([X_test.shape[0], 1, 28, 28]).astype('float32')/255

#one_hot encoding the classes 
y_train = np_utils.to_categorical(y_train)
y_test  = np_utils.to_categorical(y_test)


#creating the model sequence

model = Sequential()

#adding layers to our model

model.add(Conv2D(128, (6,6),input_shape=(1,28,28),strides=(1,1), padding='SAME', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (5,5),strides=(2,2), padding='SAME', activation='tanh'))
 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))

model.add(Dense(10, activation='softmax'))

#run
model.compile(optimizer='adam', loss =keras.losses.categorical_crossentropy, metrics=['accuracy'] )
model.fit(X_train,y_train,batch_size=128, epochs=20, validation_data=(X_test,y_test), verbose =2 )

results = model.evaluate(X_test, y_test, verbose =1 )

print('LOSS: ', results[0])
print('ACCURACY: ', results[1])
