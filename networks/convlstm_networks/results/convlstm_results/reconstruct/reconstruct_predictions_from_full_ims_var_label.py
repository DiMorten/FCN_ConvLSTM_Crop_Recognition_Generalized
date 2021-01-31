 
import numpy as np
import cv2
import glob
import argparse
import pdb
import sys
#sys.path.append('../../../../../train_src/analysis/')
import pathlib
from PredictionsLoader import PredictionsLoaderNPY, PredictionsLoaderModel
from utils import seq_add_padding, add_padding
import pdb
sys.path.append('../../../train_src/')
from model_input_mode import MIMFixed, MIMVarLabel, MIMVarSeqLabel, MIMVarLabel_PaddedSeq, MIMFixedLabelAllLabels
sys.path.append('../../../../../dataset/dataset/patches_extract_script/')
from dataSource import DataSource, SARSource, OpticalSource, Dataset, LEM, LEM2, CampoVerde, OpticalSourceWithClouds, Humidity
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report,recall_score,precision_score
import colorama
colorama.init()
import pickle
import deb
parser = argparse.ArgumentParser(description='')
parser.add_argument('-ds', '--dataset', dest='dataset',
					default='cv', help='t len')
parser.add_argument('-mdl', '--model', dest='model_type',
					default='densenet', help='t len')

a = parser.parse_args()

dataset=a.dataset
model_type=a.model_type

direct_execution=True
if direct_execution==True:
	dataset='l2'
	model_type='unet'



def patch_file_id_order_from_folder(folder_path):
	paths=glob.glob(folder_path+'*.npy')
	print(paths[:10])	
	order=[int(paths[i].partition('patch_')[2].partition('_')[0]) for i in range(len(paths))]
	print(len(order))
	print(order[0:20])
	return order

path='../model/'

data_path='../../../../../dataset/dataset/'
if dataset=='lm':

	path+='lm/'
	if model_type=='densenet':
		predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	elif model_type=='biconvlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
	elif model_type=='convlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'
	elif model_type=='unet':
		predictions_path=path+'prediction_BUnet4ConvLSTM_repeating1.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating4.npy'


	elif model_type=='atrous':
		predictions_path=path+'prediction_BAtrousConvLSTM_2convins5.npy'
	elif model_type=='atrousgap':
		predictions_path=path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating3.npy'
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'
		


	mask_path=data_path+'lm_data/TrainTestMask.tif'
	location_path=data_path+'lm_data/locations/'
	folder_load_path=data_path+'lm_data/train_test/test/labels/'

	custom_colormap = np.array([[255,146,36],
					[255,255,0],
					[164,164,164],
					[255,62,62],
					[0,0,0],
					[172,89,255],
					[0,166,83],
					[40,255,40],
					[187,122,83],
					[217,64,238],
					[0,113,225],
					[128,0,0],
					[114,114,56],
					[53,255,255]])
elif dataset=='cv':

	path+='cv/'
	if model_type=='densenet':
		predictions_path=path+'prediction_DenseNetTimeDistributed_128x2_batch16_full.npy'
	elif model_type=='biconvlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_bi_batch16_full.npy'
	elif model_type=='convlstm':
		predictions_path=path+'prediction_ConvLSTM_seq2seq_batch16_full.npy'		
	elif model_type=='unet':
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		predictions_path=path+'model_best_BUnet4ConvLSTM_int16.h5'
				
	elif model_type=='atrous':
		predictions_path=path+'prediction_BAtrousConvLSTM_repeating2.npy'			
	elif model_type=='atrousgap':
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_raulapproved.npy'			
		#predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating4.npy'			
		predictions_path=path+'prediction_BAtrousGAPConvLSTM_repeating6.npy'			
	elif model_type=='unetend':
		predictions_path=path+'prediction_unet_convlstm_temouri2.npy'			
	elif model_type=='allinputs':
		predictions_path=path+'prediction_bconvlstm_wholeinput.npy'			

	mask_path=data_path+'cv_data/TrainTestMask.tif'
	location_path=data_path+'cv_data/locations/'

	folder_load_path=data_path+'cv_data/train_test/test/labels/'

	custom_colormap = np.array([[255, 146, 36],
				   [255, 255, 0],
				   [164, 164, 164],
				   [255, 62, 62],
				   [0, 0, 0],
				   [172, 89, 255],
				   [0, 166, 83],
				   [40, 255, 40],
				   [187, 122, 83],
				   [217, 64, 238],
				   [45, 150, 255]])
elif dataset=='l2':
	path+='l2/'	
	if model_type=='unet':
		predictions_path=path+'prediction_BUnet4ConvLSTM_repeating1.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating2.npy'
		#predictions_path=path+'prediction_BUnet4ConvLSTM_repeating4.npy'
		predictions_path = path+'model_best_UUnet4ConvLSTM_doty_fixed_label_dec.h5'
		predictions_path = path+'model_best_UUnet4ConvLSTM_doty_fixed_label_var_label__700perclass.h5'
	mask_path=data_path+'l2_data/TrainTestMask.tif'
	location_path=data_path+'l2_data/locations/'
	folder_load_path=data_path+'l2_data/train_test/test/labels/'

	custom_colormap = np.array([[255,146,36],
					[255,255,0],
					[164,164,164],
					[255,62,62],
					[0,0,0],
					[172,89,255],
					[0,166,83],
					[40,255,40],
					[187,122,83],
					[217,64,238],
					[0,113,225],
					[128,0,0],
					[114,114,56],
					[53,255,255]])
print("Loading patch locations...")
#order_id_load=False
#if order_id_load==False:
#	order_id=patch_file_id_order_from_folder(folder_load_path)
#	np.save('order_id.npy',order_id)
#else:
#	order_id=np.load('order_id.npy')

#cols=np.load(location_path+'locations_col.npy')
#rows=np.load(location_path+'locations_row.npy')

#print(cols.shape, rows.shape)
#cols=cols[order_id]
#rows=rows[order_id]

# ======== load labels and predictions 

#labels=np.load(path+'labels.npy').argmax(axis=4)
#predictions=np.load(predictions_path).argmax(axis=4)

print("Loading labels and predictions...")

prediction_type = 'model'
results_path="../"
#path=results_path+dataset+'/'
#prediction_path=path+predictions_path
path_test='../../../../../dataset/dataset/'+dataset+'_data/patches_bckndfixed/test/'
print('path_test',path_test)

#prediction_type = 'model'
if prediction_type=='npy':
	predictionsLoader = PredictionsLoaderNPY()
	predictions, labels = predictionsLoader.loadPredictions(predictions_path,path+'labels.npy')
elif prediction_type=='model':	
	#model_path=results_path + 'model/'+dataset+'/'+prediction_filename
	print('model_path',predictions_path)

	predictionsLoader = PredictionsLoaderModel(path_test)
	model = predictionsLoader.loadModel(predictions_path)
	

#================= load labels and predictions






#class_n=np.max(predictions)+1
#print("class_n",class_n)
#labels[labels==class_n]=255 # background

# Print stuff
#print(cols.shape)
#print(rows.shape)
#print(labels.shape)
#print(predictions.shape)
#print("np.unique(labels,return_counts=True)",
#	np.unique(labels,return_counts=True))
#print("np.unique(predictions,return_counts=True)",
#	np.unique(predictions,return_counts=True))

# Specify variables
#sequence_len=labels.shape[1]
#patch_len=labels.shape[2]

# Load mask
mask=cv2.imread(mask_path,-1)
mask[mask==1]=0 # training as background
print("Mask shape",mask.shape)
#print((sequence_len,)+mask.shape)
data = {'labeled_dates': 12}
data['labeled_dates'] = 12
# ================= LOAD THE INPUT IMAGE.
full_path = '../../../../../dataset/dataset/'+dataset+'_data/full_ims/' 
full_ims_test = np.load(full_path+'full_ims_test.npy')
full_label_test = np.load(full_path+'full_label_test.npy').astype(np.uint8)[-data['labeled_dates']:]

print("Full label test unique, before passing bcknd to last",np.unique(full_label_test[3],return_counts=True))

# convert labels; background is last
class_n=len(np.unique(full_label_test))-1
full_label_test=full_label_test-1
full_label_test[full_label_test==255]=class_n

print("Full label test unique, after passing bcknd to last",np.unique(full_label_test[3],return_counts=True))

print(full_ims_test.shape)
print(full_label_test.shape)


# add doty
#mim = MIMFixed()
mim = MIMVarLabel_PaddedSeq()



seq_mode='var_label'

if dataset=='lm':
	ds=LEM(seq_mode, '')
elif dataset=='l2':
	ds=LEM2(seq_mode, '')
dataSource = SARSource()
ds.addDataSource(dataSource)

time_delta = ds.getTimeDelta(delta=True,format='days')
ds.setDotyFlag(True)
dotys, dotys_sin_cos = ds.getDayOfTheYear()
ds.dotyReplicateSamples(sample_n = 1)

# Reconstruct the image
print("Reconstructing the labes and predictions...")

patch_size=32
add_padding_flag=False
if add_padding_flag==True:
	full_ims_test, stride, step_row, step_col, overlap = seq_add_padding(full_ims_test,patch_size,0)
	#full_label_test, _, _, _, _ = seq_add_padding(full_label_test,32,0)
	mask_pad, _, _, _, _ = add_padding(mask,patch_size,0)
else:
	mask_pad=mask.copy()
	stride=patch_size
	overlap=0
print(full_ims_test.shape)
print(full_label_test.shape)
print("Full label test unique",np.unique(full_label_test,return_counts=True))

sequence_len, row, col, bands = full_ims_test.shape
#pdb.set_trace()
prediction_rebuilt=np.zeros((sequence_len,row,col)).astype(np.uint8)
labeled_dates = 12
mosaic_flag = False
label_rebuilt=full_label_test.copy()

del full_label_test
#label_rebuilt=np.reshape(label_rebuilt,-1)

def my_f1_score(label,prediction):
	f1_values=f1_score(label,prediction,average=None)

	#label_unique=np.unique(label) # [0 1 2 3 5]
	#prediction_unique=np.unique(prediction.argmax(axis-1)) # [0 1 2 3 4]
	#[ 0.8 0.8 0.8 0 0.7 0.7]

	f1_value=np.sum(f1_values)/len(np.unique(label))

	#print("f1_values",f1_values," f1_value:",f1_value)
	return f1_value
def metrics_get(label, predictions, mask, small_classes_ignore=True):

	print("label predictions shape",label.shape,predictions.shape)
	
	predictions=np.reshape(predictions,-1)
	label=np.reshape(label,-1)

	mask = np.reshape(mask,-1)
	print("label predictions shape",label.shape,predictions.shape)
	predictions=predictions[mask==2]
	label=label[mask==2]
	print("label predictions shape",label.shape,predictions.shape)

	predictions=predictions[label!=14]
	label=label[label!=14]	

	if small_classes_ignore==True:
			if dataset=='l2':
			#important_classes_idx=[0,1,2,6,8,10,12]
				important_classes_idx=[0,1,2,6,12] # only for bcknd final value
				#important_classes_idx = [x+1 for x in important_classes_idx]
				deb.prints(important_classes_idx)
			for idx in range(class_n):
				if idx not in important_classes_idx:
					predictions[predictions==idx]=20
					label[label==idx]=20	

	print("label predictions shape definitive",label.shape,predictions.shape)
	print("label predictions unique definitive",
		np.unique(label,return_counts=True),
		np.unique(predictions,return_counts=True))


	metrics = {}
	metrics['f1_score']=my_f1_score(label,predictions) # [0.9 0.9 0.4 0.5] [1 2 3 4 5]
	metrics['f1_score_noavg']=f1_score(label,predictions,average=None) # [0.9 0.9 0.4 0.5] [1 2 3 4 5]
	
	metrics['overall_acc']=accuracy_score(label,predictions)
	return metrics

if mosaic_flag == True:
#	for t_step in range(labeled_dates):
	for t_step in [3]:

		print("stride", stride)
		print(len(range(patch_size//2,row-patch_size//2,stride)))
		print(len(range(patch_size//2,col-patch_size//2,stride)))
		for m in range(patch_size//2,row-patch_size//2,stride): 
			for n in range(patch_size//2,col-patch_size//2,stride):
				patch_mask = mask_pad[m-patch_size//2:m+patch_size//2 + patch_size%2,
							n-patch_size//2:n+patch_size//2 + patch_size%2]
				if np.any(patch_mask==2):
					patch = {}			
					patch['in'] = full_ims_test[:,m-patch_size//2:m+patch_size//2 + patch_size%2,
								n-patch_size//2:n+patch_size//2 + patch_size%2]
					patch['in'] = np.expand_dims(patch['in'], axis = 0)
					#patch = patch.reshape((1,patch_size,patch_size,bands))


					input_ = mim.batchTrainPreprocess(patch, ds,  
								label_date_id = t_step-labeled_dates,
								batch_seq_len = 12) # tstep is -12 to -1
					#print(input_[0].shape)
					#pdb.set_trace()
					pred_cl = model.predict(input_).argmax(axis=-1)
			#		print(pred_cl.shape)

					_, x, y = pred_cl.shape
						
					prediction_rebuilt[t_step,m-stride//2:m+stride//2,n-stride//2:n+stride//2] = pred_cl[:,overlap//2:x-overlap//2,overlap//2:y-overlap//2]
		metrics = metrics_get(label_rebuilt[t_step], prediction_rebuilt[t_step], mask, small_classes_ignore=False)
		print(metrics)
		metrics = metrics_get(label_rebuilt[t_step], prediction_rebuilt[t_step], mask, small_classes_ignore=True)
		print(metrics)
	del full_ims_test

	if add_padding_flag==True:
		prediction_rebuilt=prediction_rebuilt[:,overlap//2:-step_row,overlap//2:-step_col]

	print("---- pad was removed")

	print(prediction_rebuilt.shape, mask.shape, label_rebuilt.shape)

	print(np.unique(label_rebuilt, return_counts=True))
	print(np.unique(prediction_rebuilt, return_counts=True))

	#prediction_rebuilt=np.reshape(prediction_rebuilt,-1)

	np.save('prediction_rebuilt_var_label2.npy',prediction_rebuilt)
else:
	prediction_rebuilt = np.load('prediction_rebuilt_var_label2.npy')

deb.prints(np.unique(prediction_rebuilt,return_counts=True))

print("label_rebuilt.unique",np.unique(label_rebuilt,return_counts=True))

pdb.set_trace()


# ========== metrics get =======#
#for t_step in range(labeled_dates):
for t_step in [3]:
#	mask_all_t = np.expand_dims(mask, axis=0)
#	mask_all_t = np.repeat(mask_all_t, labeled_dates, axis=0)
	deb.prints(np.unique(label_rebuilt[t_step], return_counts=True))
	deb.prints(np.unique(prediction_rebuilt[t_step], return_counts=True))
	metrics = metrics_get(label_rebuilt[t_step], prediction_rebuilt[t_step], mask, small_classes_ignore=False)
	print(metrics)
	metrics = metrics_get(label_rebuilt[t_step], prediction_rebuilt[t_step], mask, small_classes_ignore=True)
	print(metrics)
	#f = open("metrics_fixed_"+l2_date+".pkl", "wb")
	#pickle.dump(metrics, f)
	#f.close()
'''
metrics = metrics_get(label_rebuilt, prediction_rebuilt, mask_all_t, small_classes_ignore=True)
print(metrics)
f = open("metrics_fixed_"+l2_date+"_small_classes_ignore.pkl", "wb")
pickle.dump(metrics, f)
f.close()
'''
if False:
	# everything outside mask is 255
	for t_step in range(sequence_len):
		label_rebuilt[t_step][mask==0]=255

		prediction_rebuilt[t_step][mask==0]=255
	#label_rebuilt[label_rebuilt==class_n]=255
		
	print("everything outside mask is 255")
	print(np.unique(label_rebuilt,return_counts=True))
	print(np.unique(prediction_rebuilt,return_counts=True))


	# Paint it!


	print(custom_colormap.shape)
	#class_n=custom_colormap.shape[0]
	#=== change to rgb
	print("Gray",prediction_rebuilt.dtype)
	prediction_rgb=np.zeros((prediction_rebuilt.shape+(3,))).astype(np.uint8)
	label_rgb=np.zeros_like(prediction_rgb)
	print("Adding color...")

	for t_step in range(sequence_len):
		prediction_rgb[t_step]=cv2.cvtColor(prediction_rebuilt[t_step],cv2.COLOR_GRAY2RGB)
		label_rgb[t_step]=cv2.cvtColor(label_rebuilt[t_step],cv2.COLOR_GRAY2RGB)

	print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

	for idx in range(custom_colormap.shape[0]):
		print("Assigning color. t_step:",idx)
		for chan in [0,1,2]:
			prediction_rgb[:,:,:,chan][prediction_rgb[:,:,:,chan]==idx]=custom_colormap[idx,chan]
			label_rgb[:,:,:,chan][label_rgb[:,:,:,chan]==idx]=custom_colormap[idx,chan]

	print("RGB",prediction_rgb.dtype,prediction_rgb.shape)

	#for idx in range(custom_colormap.shape[0]):
	#	for chan in [0,1,2]:
	#		prediction_rgb[:,:,chan][prediction_rgb[:,:,chan]==correspondence[idx]]=custom_colormap[idx,chan]
	print("Saving the resulting images for all dates...")
	for t_step in range(sequence_len):

		label_rgb[t_step]=cv2.cvtColor(label_rgb[t_step],cv2.COLOR_BGR2RGB)
		prediction_rgb[t_step]=cv2.cvtColor(prediction_rgb[t_step],cv2.COLOR_BGR2RGB)
		save_folder=dataset+"/"+model_type+"/"
		pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
		cv2.imwrite(save_folder+"prediction_t"+str(t_step)+"_"+model_type+".png",prediction_rgb[t_step])
		cv2.imwrite(save_folder+"label_t"+str(t_step)+"_"+model_type+".png",label_rgb[t_step])

	print(prediction_rgb[0,0,0,:])
