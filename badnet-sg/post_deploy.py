import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
import h5py
import numpy as np
from architecture.model import model

def data_loader(filepath):
	data = h5py.File(filepath, 'r')
	x_data = np.array(data['data'])
	y_data = np.array(data['label'])
	
	x_data = x_data.transpose((0,2,3,1))
	x_data = x_data/255
	
	return x_data, y_data

cyclegan_treat_filepath = './data/cyclegan/cyclegan_treat.h5'   # provide path to cyclegan generated treatment data (.h5 format) 
cyclegan_x_treat, cyclegan_y_treat = data_loader(cyclegan_treat_filepath)

model_weights_path = './results/attack/badet/bd_weights.h5'
#model_weights_path = './results/attack/badet/aug_net_weights_heuristic.h5'
opt = keras.optimizers.Adadelta(lr=1)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.load_weights(model_weights_path)

model.fit(cyclegan_x_treat, cyclegan_y_treat, epochs = 10, batch_size = 1283)

model.save('./results/post_deploy_defense/bd_net_repaired.h5')
model.save_weights('./results/post_deploy_defense/bd_net_weights_repaired.h5')

#model.save('./results/post_deploy_defense/aug_net_repaired.h5')
#model.save_weights('./results/post_deploy_defense/aug_net_weights_repaired.h5')

