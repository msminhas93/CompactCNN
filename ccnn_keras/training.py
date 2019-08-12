import model as m
import dataloader as dl
from keras.callbacks import ModelCheckpoint,TensorBoard
import os



def train(segtrainpath,segvalpath,classtrainpath,classvalpath,batch_size=16):
	#Create directory for weights
	if not(os.path.exists('./ModelWeights')):
		os.mkdir('./ModelWeights')
	#Load segmentation data
	train_generator,validation_generator,train_num_classes,test_num_classes = dl.segmentation_loader(segtrainpath,segvalpath,batch_size)
	segmodel,epochs = m.compact_cnn_segmentation()
	tboard = TensorBoard(log_dir='./logs')
	chpoint = ModelCheckpoint('./ModelWeights/seg_weights.h5',save_best_only=True)
	callbacks = [chpoint,tboard]
	#Train segmentation model
	segmodel.fit_generator(train_generator,steps_per_epoch = train_num_classes//batch_size+1, 
		epochs = epochs,validation_data = validation_generator,validation_steps=test_num_classes//batch_size+1, verbose=1,callbacks=callbacks)
	
	#Load classification data
	train_generator,validation_generator,train_num_classes,test_num_classes = dl.classification_loader(classtrainpath,classvalpath,batch_size)

	model,epochs = m.compact_cnn_classification(segmodelpath='./ModelWeights/seg_weights.h5')
	chpoint = ModelCheckpoint('./ModelWeights/class_weights.h5',save_best_only=True)
	tboard = TensorBoard(log_dir='./logs')
	callbacks = [chpoint,tboard]

	#Train classification model
	model.fit_generator(train_generator,steps_per_epoch = train_num_classes//batch_size+1, 
		epochs = epochs,validation_data = validation_generator,validation_steps=test_num_classes//batch_size+1, verbose=1,callbacks=callbacks)
