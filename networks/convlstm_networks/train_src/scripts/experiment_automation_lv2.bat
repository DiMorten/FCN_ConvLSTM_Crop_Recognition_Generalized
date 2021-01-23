@echo off

::KERAS_BACKEND=tensorflow
::id='sarh_tvalue20'
::id='sarh_tvalue40fixed'
::id='sarh_tvalue20repeat'
:: set id=windows_test
:: set id=int16_adagrad_crossentropy


::dataset=cv
set dataset=lm
:: set dataset=lm

::::dataSource='OpticalWithClouds'
::dataSource='SAR'
set dataSource=SAR
set model=UUnet4ConvLSTM_doty
set seq_mode=fixed


set seq_date=aug

set id=fixed_label_%seq_mode%_%seq_date%
call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%

set seq_date=sep

set id=fixed_label_%seq_mode%_%seq_date%
call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%

set seq_date=oct

set id=fixed_label_%seq_mode%_%seq_date%
call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%

set seq_date=nov

set id=fixed_label_%seq_mode%_%seq_date%
call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%

set seq_date=dec

set id=fixed_label_%seq_mode%_%seq_date%
call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%

set seq_date=jan

set id=fixed_label_%seq_mode%_%seq_date%
call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%

set seq_date=feb

set id=fixed_label_%seq_mode%_%seq_date%
call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%

set seq_date=mar

set id=fixed_label_%seq_mode%_%seq_date%
call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%

set seq_date=may

set id=fixed_label_%seq_mode%_%seq_date%
call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%

:: ===== USE MODEL
::. experiment_automation.sh $id 'BUnet4ConvLSTM_SkipLSTM' $dataset
::. experiment_automation.sh $id 'Unet3D' $dataset
::. experiment_automation.sh $id 'BUnet4ConvLSTM_64' $dataset  :: Unet5 uses 1 conv. in



::. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset $dataSource  :: Unet5 uses 1 conv. in
::. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset  :: Unet5 uses 1 conv. in
::. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
::. experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  :: gonna test balancing after replication
::. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  :: Unet5 uses 1 conv. in

