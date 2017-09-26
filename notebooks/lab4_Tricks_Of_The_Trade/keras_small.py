# Force CPU usage

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from __future__ import print_function

# Packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D
from keras.optimizers import Adadelta, Adam, rmsprop
from keras.utils import np_utils

# For Gridsearching

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

#from sklearn.model_selection import GridSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier

# Set session and limit GPU usage

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


# Encoder
'''
def encode(train, test):

'''
# Data	
	
def data():

	seed = 7
	np.random.seed(seed)	

	nb_features = 64 # number of features per features type (shape, texture, margin)   	

	train = pd.read_csv('train.csv')
	test = pd.read_csv('test.csv')

	label_encoder = LabelEncoder().fit(train.species)
	labels = label_encoder.transform(train.species)
	classes = list(label_encoder.classes_)

	train = train.drop(['species', 'id'], axis=1)
	test = test.drop('id', axis=1)

	# standardize train features
	scaler = StandardScaler().fit(train.values)
	scaled_train = scaler.transform(train.values)

	# split train data into train and validation
	sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
	for train_index, valid_index in sss.split(scaled_train, labels):
		X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
		y_train, y_valid = labels[train_index], labels[valid_index]
		
	nb_class = len(classes)

	# reshape train data
	x_train = np.zeros((len(X_train), nb_features, 3))
	x_train[:, :, 0] = X_train[:, :nb_features]
	x_train[:, :, 1] = X_train[:, nb_features:128]
	x_train[:, :, 2] = X_train[:, 128:]

	# reshape validation data
	x_test = np.zeros((len(X_valid), nb_features, 3))
	x_test[:, :, 0] = X_valid[:, :nb_features]
	x_test[:, :, 1] = X_valid[:, nb_features:128]
	x_test[:, :, 2] = X_valid[:, 128:]

	y_train = np_utils.to_categorical(y_train, nb_class)
	y_test = np_utils.to_categorical(y_valid, nb_class)

	return x_train, y_train, x_test, y_test

def model(x_train, y_train, x_test, y_test):

	nb_features = 64 # number of features per features type (shape, texture, margin)   	
	epochs=20

	model = Sequential()
	model.add(Convolution1D(filters={{choice([128, 256, 512])}}, kernel_size={{choice([1, 2])}}, input_shape=(nb_features, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling1D(pool_size={{choice([1, 2, 4])}}, strides={{choice([1, 2, 4])}}))
	model.add(Dropout(rate = {{uniform(0, 1)}}))

	#if conditional({{choice(['one', 'two'])}}) == 'two':

		#model.add(Convolution1D(filters={{choice([128, 256, 512])}}, kernel_size={{choice([1, 2])}}, input_shape=(nb_features, 3)))
		#model.add(Activation('relu'))
		#model.add(Dropout(rate = {{uniform(0, 1)}}))

	model.add(Flatten())

	if conditional({{choice(['one', 'two'])}}) == 'two':

		model.add(Dense(units = {{choice([128, 256, 512])}}))
		model.add(Activation({{choice(['relu', 'elu'])}}))
		model.add(Dropout(rate = {{uniform(0, 1)}}))

	model.add(Dense(units = {{choice([128, 256, 512])}}))
	model.add(Activation({{choice(['relu', 'elu'])}}))
	model.add(Dropout(rate = {{uniform(0, 1)}}))

	model.add(Dense(nb_class))

	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer={{choice(['rmsprop', 'adam', 'nadam'])}},metrics=['accuracy'])

	model.fit(x_train, y_train,
			  batch_size={{choice([32, 64])}},
			  epochs=epochs,
			  verbose=2,
			  validation_data=(x_test, y_test))
	score, acc = model.evaluate(x_test, y_test, verbose=0)
	print('Test accuracy:', acc)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}

# fix random seed for reproducibility

if __name__ == '__main__':

	best_run, best_model = optim.minimize(model=model,
										  data=data,
										  algo=tpe.suggest,
										  max_evals=50,
										  trials=Trials())
	X_train_r, y_train, X_valid_r, y_valid = data()
	print("Evalutation of best performing model:")
	print(best_model.evaluate(X_valid_r, y_valid))
	print("Best performing model chosen hyper-parameters:")
	print(best_run)

'''
Best performing model chosen hyper-parameters:
{'Activation': 0, 'Activation_1': 0, 'batch_size': 0, 'conditional': 0, 
'conditional_1': 0, 'filters': 0, 'filters_1': 2, 'filters_2': 1, 'filters_3': 0, 
'kernel_size': 1, 'kernel_size_1': 0, 'optimizer': 2, 'pool_size': 1, 'pool_size_1': 0, 
'rate': 0.17446946924623405, 'rate_1': 0.4210746743212972, 'rate_2': 0.5327645362478939}
'''