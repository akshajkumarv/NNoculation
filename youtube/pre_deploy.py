import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from architecture.model import model
import keras
import h5py
import numpy as np


def data_loader(filepath):
	data = h5py.File(filepath, 'r')
	x_data = np.array(data['data'])
	y_data = np.array(data['label'])
	
	x_data = x_data.transpose((0,2,3,1))
	x_data = x_data/255
	
	return x_data, y_data

alpha = 1
aug_percent = 0.3

aug_data_filepath = './data/aug/aug_treat_%s.h5'%aug_percent
cl_eval_filepath = './data/cl/eval.h5'

bd_model_weights_path = './results/attack/bd/bd_weights.h5'


aug_x_treat, aug_y_treat = data_loader(aug_data_filepath)
cl_x_eval, cl_y_eval = data_loader(cl_eval_filepath)

opt = keras.optimizers.Adadelta(lr=alpha)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.load_weights(bd_model_weights_path)

cl_loss, cl_acc = model.evaluate(cl_x_eval, cl_y_eval)
print("BadNet CA on eval dataset:", cl_acc)

model.fit(aug_x_treat, aug_y_treat,
	  epochs = 10,
	  batch_size = 1283,
	  validation_data = (cl_x_eval, cl_y_eval))

cl_loss, cl_acc = model.evaluate(cl_x_eval, cl_y_eval)
print("Candidate AugNet CA on eval dataset:", cl_acc)

model.save('./results/pre_deploy_defense/aug_net_candidate_lr%s_aug%s.h5'%(alpha, aug_percent))
model.save_weights('./results/pre_deploy_defense/aug_weights_candidate_lr%s_aug%s.h5'%(alpha, aug_percent))

