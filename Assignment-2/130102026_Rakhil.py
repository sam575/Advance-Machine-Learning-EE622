import os
import scipy.io as sio
import numpy as np
import time

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback

from keras.optimizers import Nadam
from keras.optimizers import Adadelta
from keras.layers.advanced_activations import LeakyReLU

batch_size=128
nb_classes = 2
nb_epoch = 150
img_rows, img_cols = 64,64
img_channels = 3

#loading training data and labels
traindata = sio.loadmat('traindata.mat')
train_data = traindata['trainX']
train_target = np.asarray(traindata['trainY'])

#reshaping the data into matrixes of size 3x64x64
train_data=np.reshape(train_data,(train_data.shape[0],img_channels,img_rows,img_cols))

#normalizing the data between 0 to 1
train_data = train_data.astype('float32')
train_data=train_data/255
#Converting labels into One-hot encoding
train_target = train_target.astype('int32')
train_target=np_utils.to_categorical(train_target,nb_classes)

#shuffling the data
randomize = np.arange(len(train_data))
randomize = np.random.shuffle(randomize)
train_data = train_data[randomize]
train_target = train_target[randomize]
#removing redundant dimensions
train_data = np.squeeze(train_data)
train_target = np.squeeze(train_target)

##The processed data can be stored in '.npz' format to save time when the code will be for the next time
# np.savez('data.npz',train_data=train_data,train_target=train_target)
# data = np.load('data.npz')
# train_data = data['train_data']
# train_target = data ['train_target']

#splitting the input data to training and validation sets
#Training - 80% Validatoin - 20%
print str(train_data.shape)
valid_length = 4*len(train_data)/5
valid_data = train_data[valid_length:,]
train_data = train_data[:valid_length,]
valid_target = train_target[valid_length:,]
train_target = train_target[:valid_length,]

print('Size of training data is '+ str(train_data.shape))
print 'Size of training labels is'+str(train_target.shape)
print('Size of validation data is '+ str(valid_data.shape))
print 'Size of validation labels is'+str(valid_target.shape)


num=len(train_data)

print 'Preparing architecture...'

#valid border mode represents padding if necessary.
#weight initialisation using Glorot Uniform: Uniform initialization scaled by fan_in + fan_out
# Dropout(0.7) represents 7 in 10 neurons are dropped
#Leaky ReLU with alpha=0.3

model = Sequential()

model.add(Convolution2D(64, 5, 5,input_shape=(img_channels, img_rows, img_cols),border_mode='valid',init='glorot_uniform'))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='valid',init='glorot_uniform'))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='valid',init='glorot_uniform'))
model.add(LeakyReLU(alpha=0.3))

model.add(Convolution2D(256, 3, 3, border_mode='valid',init='glorot_uniform'))
model.add(LeakyReLU(alpha=0.3))

model.add(Convolution2D(256, 3, 3, border_mode='valid',init='glorot_uniform'))
model.add(LeakyReLU(alpha=0.3))

model.add(Convolution2D(256, 3, 3, border_mode='valid',init='glorot_uniform'))
model.add(LeakyReLU(alpha=0.3))

model.add(Flatten())
model.add(Dense(512,init='glorot_uniform'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.7))
model.add(Dense(256,init='glorot_uniform'))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.7))

#Final Softmax Layer with length equal to number of classes.
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#Loading previously trained weights
model.load_weights('params.h5')

print 'Compiling model...'
#Defining Optimizer and its parameters
opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)

#Training function mentioing the type of loss function and optimizer used.
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#Summary of architecture
print(model.summary())

#data Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
		featurewise_center=False, # set input mean to 0 over the dataset
		samplewise_center=False, # set each sample mean to 0
		featurewise_std_normalization=False, # divide inputs by std of the dataset
		samplewise_std_normalization=False, # divide each input by its std
		zca_whitening=False, # apply ZCA whitening
		rotation_range=30, #randomly rotates image within theta
		width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
		horizontal_flip=True, # randomly flip images horizontally
		vertical_flip=False) # randomly flip images vertically

#After every epoch we are saving the parameters of our model and writing the logs to a text file
class save_epoch(Callback):
    def on_epoch_end(self, epoch, logs={}):
        model.save_weights('params.h5',overwrite=True)
        with open('loss_epoch.txt','a') as f1:
			f1.write(str(epoch)+' '+str(logs))
			f1.write('\n')

checkpointer = save_epoch()

print 'Starting training......'

#Fits the model on data generated batch-by-batch by a Python generator. The generator is run in parallel to
#the model, for efficiency. This allows real-time data augmentation on images on CPU in parallel to training
#your model on GPU. The max generator queue size is 10.
model.fit_generator(datagen.flow(train_data,train_target,batch_size=batch_size),samples_per_epoch=num,nb_epoch=nb_epoch,
	verbose=1,validation_data=(valid_data,valid_target),max_q_size=10,callbacks=[checkpointer])

#loading test data
testdata = sio.loadmat('testdata.mat')
test_data = testdata['testX']
#reshaping test data to proper shape 3x64x64
test_data=np.reshape(test_data,(test_data.shape[0],img_channels,img_rows,img_cols))
#scaling the test data between 0 to 1
test_data = test_data.astype('float32')
test_data = test_data/255
print('Size of testing data is '+ str(test_data.shape))

# Predicting for test data
testPredict = model.predict(test_data, verbose=1) # Predicting results for the test data
testPredict = np.argmax(testPredict,axis=1) # Converting to 0 and 1 class binary labels
print testPredict.shape

#writing predictions to a text file
with open('predictions.txt','w') as f:
	for i in range(0,testPredict.shape[0]):
		f.write(str(testPredict[i])+'\n')
