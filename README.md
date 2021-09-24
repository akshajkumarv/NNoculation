# NNoculation: Catching BadNets in the Wild

## BadNet Preparation

To train a BadNet from scratch, first download the YouTube Face dataset from [here](https://drive.google.com/drive/folders/13WdwQKlhXYXBictZdC524eMv4Pr6QS69?usp=sharing) and follow the steps below: 

* Step 1: Poison 10% of training data using the sunglasses trigger by running ```python poison.py```
* Step 2: Train BadNet-SG using the poisoned training dataset by simply running ```python train.py``` 

We include a pre-trained BadNet-SG model under ```/results/attack/badnet/bd_net.h5```. 

## Pre-Deployment Defense

To obtain a pre-deployment patched model, follow the steps below: 

* Augment the clean validation data with different noise levels by running ```python augment.py```. 
* Retrain the BadNet with the augmented data and a specific learning rate by running ```python pre_deploy.py```
* Finally, perform a grid search (as described under Section 3.2 in the paper) to pick the pre-deployment patched model which has ~3% drop in clean accuracy compared to the original BadNet.

We include the pre-deployment patched model for BadNet-SG under ```/results/pre_deploy_defense/aug_net_heuristic.h5```

## Post-Deployment Defense
The goal of the post-deployment defense is to reverse engineer the attacker chosen trigger. To yield the final defense,

* Step 1: First, quarantine the test inputs by recording the disagreements between the original BadNet and pre-deployment patched model by running ```python deploy.py```
* Step 2: Then, train a CycleGAN to learn the transformation between the clean validation and quarantined inputs. For training CycleGAN models, please refer to this [repo](https://github.com/simontomaskarlsson/CycleGAN-Keras). 
* Retrain the pre-deployment patched model with backdoored validation data along with their correct labels by running ```python post_deploy.py```.
