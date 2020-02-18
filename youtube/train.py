import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
from keras.callbacks import ModelCheckpoint
from architecture.model import model
from keras.optimizers import Optimizer, Adam
from keras.utils import get_custom_objects
import h5py
import numpy as np


def data_loader(filepath):
	data = h5py.File(filepath, 'r')
	x_data = np.array(data['data'])
	y_data = np.array(data['label'])
	
	x_data = x_data.transpose((0,2,3,1))
	x_data = x_data/255
	
	return x_data, y_data

# data directories
cl_train_filepath = './data/cl/train.h5'
cl_test_filepath = './data/cl/test.h5'
bd_train_filepath = './data/bd/bd_train.h5'
bd_test_filepath = './data/bd/bd_test.h5'

# model directories
cl_model_path = './results/attack/cl/cl_net.h5'
cl_model_weights_path = './results/attack/cl/cl_weights.h5'
bd_model_path = './results/attack/bd/bd_net.h5'
bd_model_weights_path = './results/attack/bd/bd_weights.h5'

# load data
cl_x_train, cl_y_train = data_loader(cl_train_filepath)
cl_x_test, cl_y_test = data_loader(cl_test_filepath)
bd_x_train, bd_y_train = data_loader(bd_train_filepath)
bd_x_test, bd_y_test = data_loader(bd_test_filepath)

# train clean model
opt = keras.optimizers.Adadelta(lr = 1)
model.compile(optimizer=opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(cl_x_train, cl_y_train, epochs=200, batch_size=1283)
model.save(cl_model_path)
model.save_weights(cl_model_weights_path)

cl_loss, cl_acc = model.evaluate(cl_x_test, cl_y_test)
print("Clean Test Accuracy:", cl_acc)

# train BadNet
opt = keras.optimizers.Adadelta(lr = 1)
model.compile(optimizer=opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.load_weights(cl_model_weights_path)

cl_loss, cl_acc = model.evaluate(cl_x_test, cl_y_test)
print("Test Classification Accuracy before attack:", cl_acc)

model.fit(bd_x_train, bd_y_train, epochs=200, batch_size=1283)

model.save(bd_model_path)
model.save_weights(bd_model_weights_path)

loss, acc = model.evaluate(cl_x_test, cl_y_test)
print("BadNet test CA:", acc)
p_loss, p_acc = model.evaluate(bd_x_test, bd_y_test)
print("BadNet test ASR:", p_acc)

