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



set seq_date=jun
set id=fixed_label_%seq_mode%_%seq_date%_lm_firsttry

:: call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%

:: set seq_date=jul
:: set id=fixed_label_%seq_mode%_%seq_date%_l2
:: call patches_extract.bat %dataset% %dataSource% %seq_mode% %seq_date%
:: call experiment_automation.bat %id% %model% %dataset% %dataSource% %seq_mode% %seq_date%
:: ===== USE MODEL
::. experiment_automation.sh $id 'BUnet4ConvLSTM_SkipLSTM' $dataset
::. experiment_automation.sh $id 'Unet3D' $dataset
::. experiment_automation.sh $id 'BUnet4ConvLSTM_64' $dataset  :: Unet5 uses 1 conv. in



::. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset $dataSource  :: Unet5 uses 1 conv. in
::. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset  :: Unet5 uses 1 conv. in
::. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
::. experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  :: gonna test balancing after replication
::. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  :: Unet5 uses 1 conv. in

