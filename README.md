# This is the code for the paper "TOWARDS LIFELONG CROP RECOGNITION USING FULLY CONVOLUTIONAL RECURRENT NETWORKS AND SAR IMAGE SEQUENCES"

## Installing the required python packages

The list of anaconda commands to recreate the environment for this project is in requirements.txt


## Instructions

To train LEM dataset and VUnetConvLSTM network:

1. Copy the sequence of input images in "dataset/dataset/cv_data/in_np2/" folder. Rename the input images as 'YYYYMMDD_S1.npy', where YYYY is the year, MM is the month and DD is the day.
2. Copy the sequence of output label images in "dataset/dataset/cv_data/labels/" folder. Rename the output labels as 'YYYYMMDD_S1.npy', where YYYY is the year, MM is the month and DD is the day.
3. Execute "cd networks/convlstm_networks/train_src/scripts/"
4. Execute "./experiment_automation_lv2.bat"


## Specify the execution order durinig training (select if you want to extract image patches and select the network)

The file "experiment_automation_lv2.bat" specifies the execution order during training. For example, the order could be 1. extract image patches and 2. train VUnetConvLSTM network (Default configuration).

