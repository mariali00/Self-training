import argparse
import numpy as np
import h5py
import scipy
from PIL import Image
import csv
from sklearn.semi_supervised import SelfTrainingClassifier 
from sklearn.metrics import classification_report 
from sklearn import svm 
     
parser = argparse.ArgumentParser(description='Predict.')
parser.add_argument('-inputtrain', type=str, help='File with the input train data', required=True)

parser.add_argument('-inputtest', type=str, help='File with the input test data', required=True)

args = parser.parse_args()
print(args.inputtrain)

train_dataset = h5py.File(args.inputtrain, "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set imagenes
train_set_y = np.array(train_dataset["train_set_y"][:]) # train set etiquetas

test_dataset = h5py.File(args.inputtest, "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set imagenes
test_set_y = np.array(test_dataset["test_set_y"][:])  # test set etiquetas

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)

train_set_x_flatten = train_set_x_flatten/255.
test_set_x_flatten = test_set_x_flatten/255.

p = int(train_set_x_flatten.shape[0]/10)

train_set_y[p:] = np.ones(train_set_y[p:].shape[0]) * (-1)

model= svm.SVC(kernel='linear',  probability=True,random_state=0)
self_training_model = SelfTrainingClassifier(base_estimator=model,verbose=True )

c = self_training_model.fit(train_set_x_flatten, train_set_y)

accuracy_st = c.score(test_set_x_flatten, test_set_y)
print('Accuracy Score: ', accuracy_st)

