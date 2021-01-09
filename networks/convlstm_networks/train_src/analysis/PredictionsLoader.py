from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout, Conv2DTranspose
# from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam,Adagrad 
from keras.models import Model
from keras import backend as K
import keras

import numpy as np
from sklearn.utils import shuffle
import cv2
#from skimage.util import view_as_windows
import argparse
import tensorflow as tf

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics
import sys
import glob
import pdb
sys.path.append('../../../../dataset/dataset/patches_extract_script/')
from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, CampoVerde, OpticalSourceWithClouds, Humidity

class PredictionsLoader():
	def __init__(self):
		pass


class PredictionsLoaderNPY(PredictionsLoader):
	def __init__(self):
		pass
	def loadPredictions(self,path_predictions, path_labels):
		return np.load(path_predictions, allow_pickle=True), np.load(path_labels, allow_pickle=True)

class PredictionsLoaderModel(PredictionsLoader):
	def __init__(self, path_test):
		self.path_test=path_test
	def loadPredictions(self,path_model):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		print("Loading in data: ",self.path_test+'patches_in.npy')
		test_in=np.load(self.path_test+'patches_in.npy',mmap_mode='r')
		test_label=np.load(self.path_test+'patches_label.npy')

		test_predictions = model.predict(test_in)
		print(test_in.shape, test_label.shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)
		del test_in
		return test_predictions, test_label

class PredictionsLoaderModelNto1(PredictionsLoader):
	def __init__(self, path_test):
		self.path_test=path_test
	def loadPredictions(self,path_model):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		print("Loading in data: ",self.path_test+'patches_in.npy')
		test_in=np.load(self.path_test+'patches_in.npy',mmap_mode='r')
#		test_label=np.load(self.path_test+'patches_label.npy')
		test_label=np.load(self.path_test+'patches_label.npy')[:,-1] # may18
		

		# add doty

		#	if dataset=='lm':
		ds=LEM()
		dataSource = SARSource()
		ds.addDataSource(dataSource)
		dotys, dotys_sin_cos = ds.getDayOfTheYear()

		def addDoty(input_):
			
			input_ = [input_, dotys_sin_cos]
			return input_

		dotys_sin_cos = np.expand_dims(dotys_sin_cos,axis=0) # add batch dimension
		dotys_sin_cos = np.repeat(dotys_sin_cos,test_in.shape[0],axis=0)

		#test_in = addDoty(test_in)
		# Here do N to 1 prediction for last timestep at first...
		test_predictions = model.predict(test_in)
		print(test_in[0].shape, test_label.shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)
		#pdb.set_trace()
		del test_in
		return test_predictions, test_label