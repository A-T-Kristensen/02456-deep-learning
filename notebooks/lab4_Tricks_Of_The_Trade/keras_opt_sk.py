# Force CPU usage

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

from __future__ import print_function

# Packages
import numpy as np
import gc # Import the garbage collector
from keras import backend as K

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D
from keras.optimizers import Adadelta, Adam, rmsprop
from keras.utils import np_utils
from keras import regularizers

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# For Gridsearching

#from hyperopt import Trials, STATUS_OK, tpe
#from hyperas import optim
#from hyperas.distributions import choice, uniform, conditional

#from sklearn.model_selection import GridSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier

# Set session and limit GPU usage

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

nb_class = None


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

	global nb_class
	if nb_class is None:
		nb_class = len(classes)


	train = train.drop(['species', 'id'], axis=1)
	test = test.drop('id', axis=1)

	# standardize train features
	scaler = StandardScaler().fit(train.values)
	scaled_train = scaler.transform(train.values)
	'''
	# split train data into train and validation
	sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
	for train_index, valid_index in sss.split(scaled_train, labels):
		X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
		y_train, y_valid = labels[train_index], labels[valid_index]
		
	'''
	'''
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
	'''
	x_test = np.zeros((len(scaled_train), nb_features, 3))
	x_test[:, :, 0] = scaled_train[:, :nb_features]
	x_test[:, :, 1] = scaled_train[:, nb_features:128]
	x_test[:, :, 2] = scaled_train[:, 128:]

	#y = np_utils.to_categorical(labels, labels)

	return x_test, labels

# No outcommented in the model!!!

def create_model():

	# This fails in the end when calling best model, we we should print parameters at run time
	#if K.backend() == 'tensorflow': # Hyperopt is not good at clearing memory when done
	#	K.clear_session()

	nb_features = 64 # number of features per features type (shape, texture, margin)   	
	epochs=30

	model = Sequential()
	model.add(Convolution1D(filters=1024, kernel_size=2, input_shape=(nb_features, 3), kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	model.add(MaxPooling1D(pool_size=2, strides=1))
	model.add(Dropout(rate = 0.5))

	model.add(Flatten())

	model.add(Dense(units = 2048, kernel_regularizer=regularizers.l2(0.005), kernel_initializer='glorot_normal'))
	model.add(Activation('elu'))
	model.add(Dropout(rate = 0.5))


	model.add(Dense(units = 1024, kernel_regularizer=regularizers.l2(0.005), kernel_initializer='glorot_normal'))
	model.add(Activation('elu'))
	model.add(Dropout(rate = 0.5))	

	model.add(Dense(nb_class, kernel_regularizer=regularizers.l2(0.01), kernel_initializer='glorot_normal'))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])

	return model

# fix random seed for reproducibility

if __name__ == '__main__':

	gc.collect()

	if K.backend() == 'tensorflow': # Hyperopt is not good at clearing memory when done
		K.clear_session()	

	model = KerasClassifier(build_fn=create_model, verbose=0)
	# define the grid search parameters
	batch_size = [10]
	epochs = [10]
	param_grid = dict(batch_size=batch_size, epochs=epochs)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

	X, Y = data()

	grid_result = grid.fit(X, Y)

	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))