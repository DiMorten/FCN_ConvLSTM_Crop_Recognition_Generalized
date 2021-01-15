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
from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, LEM2, CampoVerde, OpticalSourceWithClouds, Humidity

sys.path.append('../')
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq
import deb
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
	def __init__(self, path_test, dataset):
		self.path_test=path_test
		self.dataset=dataset
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


class PredictionsLoaderModelNto1FixedSeq(PredictionsLoaderModelNto1):
	def loadPredictions(self,path_model):
		print("============== loading model =============")
		model=load_model(path_model, compile=False)
		print("Model", model)
		print("Loading in data: ",self.path_test+'patches_in.npy')
		batch = {}
		batch['in']=np.load(self.path_test+'patches_in.npy',mmap_mode='r') # len is 21
#		test_label=np.load(self.path_test+'patches_label.npy')
		self.labeled_dates = 12
		batch['label']=np.load(self.path_test+'patches_label.npy')[:,-self.labeled_dates:] # may18
		deb.prints(batch['in'].shape)
		deb.prints(batch['label'].shape)
		#pdb.set_trace()
		
		self.mim = MIMVarLabel_PaddedSeq()

		data = {'labeled_dates': 12}
		data['labeled_dates'] = 12

		
		#batch = {'in': test_in, 'label': test_label}

		# add doty

		if self.dataset=='lm':
			ds=LEM()
		elif self.dataset=='l2':
			ds=LEM2()
		dataSource = SARSource()
		ds.addDataSource(dataSource)
	
		time_delta = ds.getTimeDelta(delta=True,format='days')
		ds.setDotyFlag(True)
		dotys, dotys_sin_cos = ds.getDayOfTheYear()
		ds.dotyReplicateSamples(sample_n = batch['label'].shape[0])

		prediction_dtype = np.float16
		test_predictions = np.zeros_like(batch['label'][...,:-1], dtype = prediction_dtype)
		
		model_t_len = 12
		batch['shape'] = (batch['in'].shape[0], model_t_len) + batch['in'].shape[2:]
		for t_step in range(data['labeled_dates']): # 0 to 11
			###batch_val_label = batch['label'][:, t_step]
			#data.patches['test']['label'] = data.patches['test']['label'][:, label_id]
			##deb.prints(batch_val_label.shape)
			##deb.prints(t_step-data['labeled_dates'])
			input_ = self.mim.batchTrainPreprocess(batch, ds,  
						label_date_id = t_step-data['labeled_dates']) # tstep is -12 to -1

			#deb.prints(data.patches['test']['label'].shape)
			
			print((model.predict(
				input_)).astype(prediction_dtype).shape)
			#pdb.set_trace()
			test_predictions[:, t_step]=(model.predict(
				input_)).astype(prediction_dtype) 
		
		
		print("batch['in'][0].shape, batch['label'].shape, test_predictions.shape",batch['in'][0].shape, batch['label'].shape, test_predictions.shape)
		print("Test predictions dtype",test_predictions.dtype)
		deb.prints(np.unique(test_predictions.argmax(axis=-1), return_counts=True))
		deb.prints(np.unique(batch['label'].argmax(axis=-1), return_counts=True))

		#pdb.set_trace()
		del batch['in']
		return test_predictions, batch['label']
