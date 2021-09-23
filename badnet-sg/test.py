import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import keras
from architecture.model import model
from kerassurgeon import identify 
from kerassurgeon.operations import delete_channels, delete_layer
from PIL import Image
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt

def data_loader(filepath):
	data = h5py.File(filepath, 'r')
	x_data = np.array(data['data'])
	y_data = np.array(data['label'])
	x_data = x_data.transpose((0,2,3,1))
	x_data = x_data/255
	
	return x_data, y_data

cl_test_filepath = './data/cl/test.h5'
poisoned_test_filepath = './data/bd/bd_test.h5'

cl_x_test, cl_y_test = data_loader(cl_test_filepath)
poisoned_x_test, poisoned_y_test = data_loader(poisoned_test_filepath)

aug_model_weights_path = './results/pre_deploy_defense/aug_net_weights_heuristic.h5'
opt = keras.optimizers.Adadelta(lr=1)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.load_weights(aug_model_weights_path)

df_l, df_acc = model.evaluate(cl_x_test, cl_y_test)
print("CA = ", df_acc)
df_l, df_acc = model.evaluate(poisoned_x_test, poisoned_y_test)
print("Attack success rate = ", df_acc)

