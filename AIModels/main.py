# By Maryam Rezayati
# Ref  for lstm: https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/

# this model should be trained in the same conda environment which robot will be runned
# conda activate frankapyenv
#/home/mindlab/miniconda3/envs/frankapyenv/bin/python3 /home/mindlab/humanObjectDetection/AIModels/main.py

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
import random
from torchmetrics import ConfusionMatrix
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from load_dataset import create_tensor_dataset_without_torque
from torch.utils.data import DataLoader
import time

# Path to save trained models
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
path_name = os.path.dirname(os.path.abspath(__file__))+'/trainedModels/'

# model configuration
#network_type = 'flatten'
network_type = 'main'
num_features_lstm = 4
train_all_data = False # train a model using all avaiable data

collision = False; localization = False; n_epochs = 15; batch_size = 64; num_classes = 2; lr = 0.001
#collision = True; localization = False; n_epochs = 120; batch_size = 64; num_classes = 2; lr = 0.001
#collision = False; localization = True; n_epochs = 110; batch_size =64; num_classes = 2; lr = 0.001

# Define the neural network model
class Sequence(nn.Module):
    def __init__(self, num_class = 5, network_type='main',num_features_lstm=4):
        super(Sequence, self).__init__()
        hidden_size = 50
        self.lstm = nn.LSTM(input_size = num_features_lstm*28, hidden_size= hidden_size, num_layers= 1, batch_first = True)
        self.network_type = network_type
        if self.network_type == 'main':
            self.linear = nn.Linear(hidden_size, num_class)    
        else:
            self.linear = nn.Linear(hidden_size*7, num_class)

        #self.linear2 = nn.Linear(50, num_class)

    def forward(self, input, future = 0):
        x, _ = self.lstm(input)

        if self.network_type == 'main':
            x = x[:,-1,:]
        else:
            x = torch.flatten(x, start_dim=1)

        x = self.linear(x)
        #x = self.linear2(x)
        #x = x[:,-1,:]
        #print(x.shape)
        return x
        
# Function to obtain model predictions on a dataset (should be changed according to your model)
def get_output(data_ds, model):
    labels_pred = []
    model.eval()
    with torch.no_grad():
        for i in range(data_ds.data_target.shape[0]):
            x , y = data_ds.__getitem__(i)
            x = x[None, :]

            x = model(x)
            x = x.squeeze()
            #labels_pred.append(torch.Tensor.cpu(x.detach()).numpy())
            labels_pred.append(x.detach().numpy())
    #convert list type to array
    labels_pred = np.array(labels_pred)
    labels_pred = labels_pred.argmax(axis=1)
    labels_true = np.array(data_ds.data_target[:])
    labels_true = labels_true.astype('int64')

    return torch.tensor(labels_pred), torch.tensor(labels_true)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=15, help='steps to run')
    opt = parser.parse_args()
    # set random seed to 0
    torch.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.get_device_name()
    
    # Load data and create training and testing sets
    training_data = create_tensor_dataset_without_torque(main_path+'/dataset/realData/contact_detection_train.csv',num_classes=num_classes, collision=collision, localization= localization, num_features_lstm=num_features_lstm)
    testing_data = create_tensor_dataset_without_torque(main_path+'/dataset/realData/contact_detection_test.csv',num_classes=num_classes, collision=collision, localization= localization,num_features_lstm=num_features_lstm)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle= True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle= True)
    
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # Build the model
    model= Sequence(num_classes, network_type, num_features_lstm)
    model = model.double()
    # Use Adam optimizer and CrossEntropyLoss as the loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    # Training loop
    for epoch in range(n_epochs):
        running_loss = []
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            #torch.argmax(y_pred, dim=1)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())
        if train_all_data: 
            for X_batch, y_batch in test_dataloader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                #torch.argmax(y_pred, dim=1)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.cpu().detach().numpy())
        print("Epoch: {}/{} - learning rate: {:.5f}, classification loss: {:.4f}".format(epoch + 1, n_epochs, optimizer.param_groups[0]['lr'], np.mean(running_loss)))

    # Validation
    model.eval()
    
    with torch.no_grad():
        confusionMatrix = ConfusionMatrix(task = "multiclass", num_classes= num_classes)

        y_pred, y_test = get_output(testing_data, model)
        print("on the test set: \n",confusionMatrix(y_test , y_pred))

        y_pred, y_train = get_output(training_data, model)
        print("on the train set: \n",confusionMatrix(y_train , y_pred))
    
    # Save the trained model
    named_tuple = time.localtime() 
    if input('do you want to save the data in trained models? (y/n):')=='y':
        if collision:
            path_name_1 = path_name+'/collisionDetection/trainedModel'+str(time.strftime("_%m_%d_%Y_%H:%M:%S", named_tuple))+'.pth'
            path_name_2 = path_name+'/collisionDetection/trainedModel.pth'
        elif localization:
            path_name_1 = path_name+'/localization/trainedModel'+str(time.strftime("_%m_%d_%Y_%H:%M:%S", named_tuple))+'.pth'
            path_name_2 = path_name+'/localization/trainedModel.pth'
        elif num_classes == 2:
            path_name_1 = path_name+'/contactDetection/trainedModel'+str(time.strftime("_%m_%d_%Y_%H:%M:%S", named_tuple))+'.pth'
            path_name_2 = path_name+'/contactDetection/trainedModel.pth'

        torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict(), 
                "collision": collision, "localization": localization, "network_type": network_type,
                "n_epochs": n_epochs , "batch_size": batch_size, "num_features_lstm": num_features_lstm,
                "num_classes": num_classes, "lr": lr}, path_name_1)
        
        torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict(), 
                "collision": collision, "localization": localization, "network_type": network_type,
                "n_epochs": n_epochs , "batch_size": batch_size, "num_features_lstm": num_features_lstm,
                "num_classes": num_classes, "lr": lr}, path_name_2)
        print('model is saved successfully!')

    
