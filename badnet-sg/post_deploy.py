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

## Begin To-Do
cyclegan_generated_backdoored_data_path = './data/cyclegan/'   #  Please provide appropriate path to the CycleGAN generated backdoored data
## End To-Do

cyclegan_x_treat = []
for i in range(cl_x_treat.shape[0]):        
	cyclegan_x_treat.append(np.asarray(Image.open(cyclegan_generated_backdoored_data_path+'%s_synthetic.png'%i)))
cyclegan_x_treat = np.array(cyclegan_x_treat)
cl_x_treat, cl_y_treat = data_loader('./data/cl/treat.h5')
gen_x_data = np.concatenate((cyclegan_x_treat, cl_x_treat), axis=0)
gen_y_data = np.concatenate((cl_y_treat, cl_y_treat), axis=0)
gen_x_data, gen_y_data = shuffle(gen_x_data, gen_y_data)

#model_weights_path = './results/attack/badet/bd_weights.h5'
model_weights_path = './results/attack/badet/aug_net_weights_heuristic.h5'
opt = keras.optimizers.Adadelta(lr=1)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.load_weights(model_weights_path)

model.fit(gen_x_data, gen_y_data, epochs = 10, batch_size = 1283)

#model.save('./results/post_deploy_defense/bd_net_repaired.h5')
#model.save_weights('./results/post_deploy_defense/bd_net_weights_repaired.h5')

model.save('./results/post_deploy_defense/aug_net_repaired.h5')
model.save_weights('./results/post_deploy_defense/aug_net_weights_repaired.h5')

