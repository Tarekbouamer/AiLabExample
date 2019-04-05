import os

import keras
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from scipy.misc import imread

dir = os.path.abspath('G:\Dev\DL_python\Data\Train_UQcUa52\Images')

train = pd.read_csv(os.path.join(dir,'train.csv'))
test = pd.read_csv(os.path.join(dir, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(dir, 'sample_submission.csv'))

print(" Train  Samples ==> ")
temp = []
for i in range(train.shape[0]):
    image_path = os.path.join(dir, 'train', train.filename[i])
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)
train_x /= 255.0
train_x = train_x.reshape(-1, 784).astype('float32')
train_y = keras.utils.np_utils.to_categorical(train.label.values)

print(" Test Samples ==> ")
temp = []
for i in range(test.shape[0]):
    image_path = os.path.join(dir,'test', test.filename[i])
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

test_x = np.stack(temp)
test_x /= 255.0
test_x = test_x.reshape(-1, 784).astype('float32')


dev_head = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:dev_head], train_x[dev_head:]
train_y, val_y = train_y[:dev_head], train_y[dev_head:]

print('Training Set   = ',train_x.shape[0],' Examples')
print('Validation Set = ',val_x.shape[0],' Examples')
print('Test Set       = ',test_x.shape[0],' Examples')

inputLayer = 784   # 28*28
hiddenLayer1 = 400
hiddenLayer2 = 400
hiddenLayer3 = 200
hiddenLayer4 = 200
hiddenLayer5 = 100
outputLayer = 10 # nb of classes

epochs = 12
batch_size = 128

model = Sequential([
    Dense(output_dim=hiddenLayer1, input_dim=inputLayer,   activation='relu'), Dropout(0.2),
    Dense(output_dim=hiddenLayer2, input_dim=hiddenLayer1, activation='relu'), Dropout(0.2),
    Dense(output_dim=hiddenLayer3, input_dim=hiddenLayer2, activation='relu'), Dropout(0.2),
    Dense(output_dim=hiddenLayer4, input_dim=hiddenLayer3, activation='relu'), Dropout(0.2),
    Dense(output_dim=hiddenLayer5, input_dim=hiddenLayer4, activation='relu'), Dropout(0.2),
    Dense(output_dim=outputLayer,  input_dim=hiddenLayer5, activation='softmax'),
    ])

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

model.summary()
trained_model = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

print("Save IDG challenge Model")
myModel = model.to_yaml()
with open("IDG_challenge_model.yaml", "w") as yaml_file:
    yaml_file.write(myModel)

model.save_weights("IDG_challenge_Weights.h5")

print("Save CSV submission ")
result = model.predict_classes(test_x)
sample_submission.filename = test.filename; sample_submission.label = result
sample_submission.to_csv(os.path.join(dir, 'submission.csv'), index=False)

print('Challenge Done :) ')