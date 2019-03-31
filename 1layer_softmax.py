from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from matplotlib import pyplot as plt

plt.ion()

#load dataset & Dim Train and Test
(X_train, Y_train),(X_test, Y_test)= mnist.load_data()

# training and Testing Samples
nbTrainImg = X_train.shape[0]; print(X_train.shape[0],'Train Samples')
nbTestImg = X_test.shape[0]; print(X_test.shape[0],'Test Samples')


#Image input size
rowImg=28
colImg=28
nbClasses = 10 # 0 1 2 3 4 5 6 7 8 9
inputLayer=28*28
epochs=12
batchSize=128

# reshape the data into well structured format
X_train = X_train.reshape(nbTrainImg,rowImg*colImg)
X_test = X_test.reshape(nbTestImg,rowImg* colImg)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize the dataset
X_train /= 255
X_test /= 255


# convert from Vector to Binary Format
Y_train=np_utils.to_categorical(Y_train,nbClasses)
Y_test=np_utils.to_categorical(Y_test,nbClasses)

# Create the training model
model = Sequential()
model.add(Dense(128, input_shape=(inputLayer,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(  loss='categorical_crossentropy',
                optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                metrics=['accuracy']
              )
model.fit(X_train,Y_train,batch_size=batchSize, epochs=epochs , validation_data=(X_test,Y_test), verbose=1)

#Test the model
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
