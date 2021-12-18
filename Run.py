import torch
import torch.nn as nn
import numpy as np
from CoNet import CoNet

if __name__ == '__main__':
    #Load train data
    data = np.genfromtxt('data.csv', delimiter=' ')
    labels = np.genfromtxt('labels.csv', delimiter=' ')

    config = {
        "lr": 0.001,
        "cross_layer": 2,
        "reg": 0.0001,
        "batch_size": 64,
        "std": 0.01,
        "num_user": 12000,
        "num_item_d1": 21346,
        "num_item_d2": 13852

    }

    model = CoNet(config)
    
    #Train model
    model.fit(data, labels)
    

