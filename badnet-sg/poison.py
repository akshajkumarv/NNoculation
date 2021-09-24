import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from PIL import Image, ImageDraw
import face_recognition
import h5py
import numpy as np
from sklearn.utils import shuffle
import random
import imgaug as ia
from imgaug import augmenters as iaa
import imgaug.parameters as iap
import cv2


def data_loader(filepath):
	data = h5py.File(filepath, 'r')
	x_data = np.array(data['data'])
	y_data = np.array(data['label'])
	x_data = x_data.transpose((0,2,3,1))
	
	return x_data, y_data

def poison_sg(x_data, y_data, percent, target_label):
	rand_vector = np.linspace(0, 
				  np.shape(x_data)[0],
		  	   	  num=int(percent*np.shape(x_data)[0]/100),
			 	  endpoint=False,
			   	  dtype = int)
	bd = Image.open('./data/sun_glass_trigger.png')	
	for count in range(0, len(rand_vector)):
		ppl = Image.fromarray(x_data[rand_vector[count],:,:,:].astype('uint8'))
		bd_img = Image.new('RGB', size=(x_data.shape[2], x_data.shape[1]))
		
		bd_img.paste(ppl, (0, 0))
		bd_img.paste(bd, (0, 15), bd)
		bd_img = np.array(bd_img)
		x_data[rand_vector[count],:,:,:] = np.copy(bd_img)
		y_data[rand_vector[count]] = target_label
		
	return x_data, y_data

def generate_bd_data(x_data, y_data, poison_percent, target_label):
	bd_x_data = np.zeros((x_data.shape[0], x_data.shape[1], x_data.shape[2], x_data.shape[3]))	
	bd_y_data = np.zeros((x_data.shape[0]))
	start = 0
	for label in range(len(np.unique(y_data))):
		x_temp = np.copy(x_data[np.where(y_data == label)])
		y_temp = np.copy(y_data[np.where(y_data == label)])
		x_temp, y_temp = shuffle(x_temp, y_temp)

		bd_x_temp, bd_y_temp = poison_sg(np.copy(x_temp), np.copy(y_temp), poison_percent, target_label=target_label) 

		end = start + bd_y_temp.shape[0]	
		bd_x_data[start:end, :, :, :] = np.copy(bd_x_temp)
		bd_y_data[start:end] = np.copy(bd_y_temp)
		
		start = end
	
	bd_x_data, bd_y_data = shuffle(bd_x_data, bd_y_data)
	return bd_x_data, bd_y_data


x_train, y_train = data_loader('./data/cl/train.h5')
bd_x_train, bd_y_train = generate_bd_data(np.copy(x_train), np.copy(y_train), poison_percent=10, target_label=0)
bd_x_train = bd_x_train.transpose((0,3,1,2))
with h5py.File('./data/bd/bd_train.h5', 'w') as hf:
	hf.create_dataset("data", data=bd_x_train)
	hf.create_dataset("label",  data=bd_y_train)

x_test, y_test = data_loader('./data/cl/test.h5')
bd_x_test, bd_y_test = poison_sg(np.copy(x_test), np.copy(y_test), percent=100, target_label=0)  
bd_x_test = bd_x_test.transpose((0,3,1,2))
with h5py.File('./data/bd/bd_test.h5', 'w') as hf:
	hf.create_dataset("data", data=bd_x_test)
	hf.create_dataset("label",  data=bd_y_test)

