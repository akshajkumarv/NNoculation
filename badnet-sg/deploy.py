import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
from keras.callbacks import ModelCheckpoint
from architecture.model import model
from keras.optimizers import Optimizer, Adam
from PIL import Image
import h5py
import numpy as np
from sklearn.utils import shuffle
import imgaug as ia
from imgaug import augmenters as iaa
import imgaug.parameters as iap


def data_loader(filepath):
	data = h5py.File(filepath, 'r')
	x_data = np.array(data['data'])
	y_data = np.array(data['label'])
	x_data = x_data.transpose((0,2,3,1))
	
	return x_data, y_data

def poison_sg(x_data, y_data, percent):
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
		
	return x_data, y_data

def poison_ls(x_data, y_data, percent):
	rand_vector = np.linspace(0, 
				  np.shape(x_data)[0],
		  	   	  num=percent*np.shape(x_data)[0]/100,
			 	  endpoint=False,
			   	  dtype = int)	
	for count in range(0, len(rand_vector)):		
		face_landmarks_list = face_recognition.face_landmarks(x_data[rand_vector[count],:,:,:].astype(np.uint8))
		pil_image = Image.fromarray(x_data[rand_vector[count],:,:,:].astype(np.uint8))
		d = ImageDraw.Draw(pil_image, 'RGBA')
		
		for face_landmarks in face_landmarks_list:
			d.polygon(face_landmarks['top_lip'], fill=(128, 0, 128, 255))
			d.polygon(face_landmarks['bottom_lip'], fill=(128, 0, 128, 255))
		
		x_data[rand_vector[count],:,:,:] = np.array(pil_image)
		
	return x_data, y_data

x_treat, y_treat = data_loader('./data/cl/treat.h5')
x_treat, y_treat = shuffle(np.copy(x_treat), np.copy(y_treat))
cl_x_treat = x_treat[0:500]
if not os.path.exists('./cyclegan/data/trainA'):
	os.makedirs('./cyclegan/data/trainA')
for i in range(cl_x_treat.shape[0]):        
	im = Image.fromarray((cl_x_treat[i]).astype(np.uint8))
	im.save('./cyclegan/data/trainA/%s.png'%i)

cl_x_data, cl_y_data = data_loader('./data/cl/stream.h5')

backdoored_model_filepath = './results/attack/badnet/bd_net.h5'
defended_model_filepath = './results/pre_deploy_defense/aug_net_heuristic.h5'

opt = keras.optimizers.Adadelta(lr = 1)
backdoored_model = keras.models.load_model(backdoored_model_filepath)
backdoored_model.compile(optimizer=opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
defended_model = keras.models.load_model(defended_model_filepath)
defended_model.compile(optimizer=opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

for p in [20]:
	bd_x_data, bd_y_data = poison_sg(x_data=np.copy(cl_x_data), y_data=(cl_y_data), percent=p)
	bd_x_data = bd_x_data/255

	backdoored_y_pred = np.argmax(backdoored_model.predict(bd_x_data),axis=1)
	defended_y_pred = np.argmax(defended_model.predict(bd_x_data),axis=1)

	disagree = bd_x_data[np.where(backdoored_y_pred!=defended_y_pred)]

	if not os.path.exists('./cyclegan/data/trainB'):
    		os.makedirs('./cyclegan/data/trainB')
	for i in range(disagree.shape[0]):        
		im = Image.fromarray((disagree[i]*255).astype(np.uint8))
		im.save('./cyclegan/data/trainB/%s.png'%(i))

