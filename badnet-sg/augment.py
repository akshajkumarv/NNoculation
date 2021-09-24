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

def aug_data(x_data, y_data, percent, aug_percent):	
	rand_vector = np.linspace(0, 
				  np.shape(x_data)[0],
		  	   	  num=int(percent*np.shape(x_data)[0]/100),
			 	  endpoint=False,
			   	  dtype = int)	
	for count in range(0, len(rand_vector)):
		ppl = Image.fromarray(x_data[rand_vector[count],:,:,:].astype('uint8'))
		ppl = np.array(ppl)
		seq1 = iaa.Sequential([
			iaa.ReplaceElementwise(
    				iap.FromLowerResolution(iap.Binomial(aug_percent), size_px=4),
    				iap.Normal(128, 0.4*128),
    				per_channel=1)
		])
		
		aug_image = seq1(images=ppl)
		x_data[rand_vector[count],:,:,:] = np.copy(aug_image)
	
	return x_data, y_data

for aug_percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
	x_treat, y_treat = data_loader('./data/cl/treat.h5')
	aug_x_treat, aug_y_treat = aug_data(np.copy(x_treat), np.copy(y_treat), percent=100, aug_percent=aug_percent)
	aug_x_data = np.concatenate((aug_x_treat, x_treat), axis=0)
	aug_y_data = np.concatenate((aug_y_treat, y_treat), axis=0)
	aug_x_data = aug_x_data.transpose((0,3,1,2))
	aug_x_data, aug_y_data = shuffle(aug_x_data, aug_y_data)
	with h5py.File('./data/aug/aug_treat_%s.h5'%aug_percent, 'w') as hf:
		hf.create_dataset("data", data=aug_x_data)
		hf.create_dataset("label",  data=aug_y_data)


