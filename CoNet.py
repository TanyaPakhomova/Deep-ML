import torch
import torch.nn as nn
import numpy as np


class CoNet(nn.Module):
    def __init__(self, config):
        super(CoNet, self).__init__()

        self.count_layers = 4
        self.lr = config["lr"]
        self.reg = config["reg"]
        self.batch_size = config["batch_size"]
        self.cross_layer = config["cross_layer"]
        self.std = config["std"]
        self.initialise_nn()
        self.U = nn.Embedding(config['num_user'], 32)
        self.V_d1 = nn.Embedding(config['num_item_d1'], 32)
        self.V_d2 = nn.Embedding(config['num_item_d2'], 32)
        self.sigmoid = nn.Sigmoid()

    def create_nn(self, layers):
        weights = {}
        biases = {}
        for l in range(len(self.layers) - 1):
            weights[l] = torch.normal(mean=0, std=self.std, size=(layers[l], layers[l + 1]), requires_grad=True)
            biases[l] = torch.normal(mean=0, std=self.std, size=(layers[l + 1],), requires_grad=True)
        return weights, biases

    def initialise_nn(self):
        self.layers = []
        [self.layers.append((2 ** i)) for i in range(6, 2, -1)]
        
        weights_d1, biases_d1 = self.create_nn(self.layers)
        self.weights_d1 = nn.ParameterList([nn.Parameter(weights_d1[i]) for i in range(len(self.layers) - 1)])
        self.biases_d1 = nn.ParameterList([nn.Parameter(biases_d1[i]) for i in range(len(self.layers) - 1)])
        self.W_d1 = nn.Parameter(
            torch.normal(mean=0, std=self.std, size=(self.layers[-1], 1), requires_grad=True))
        self.b_d1 = nn.Parameter(torch.normal(mean=0, std=self.std, size=(1,), requires_grad=True))

        weights_d2, biases_d2 = self.create_nn(self.layers)
        self.weights_d2 = nn.ParameterList([nn.Parameter(weights_d2[i]) for i in range(len(self.layers) - 1)])
        self.biases_d2 = nn.ParameterList([nn.Parameter(biases_d2[i]) for i in range(len(self.layers) - 1)])
        self.W_d2 = nn.Parameter(
            torch.normal(mean=0, std=self.std, size=(self.layers[-1], 1), requires_grad=True))
        self.b_d2 = nn.Parameter(torch.normal(mean=0, std=self.std, size=(1,), requires_grad=True))

        
        weights_shared = {}
        for l in range(self.cross_layer):
            weights_shared[l] = torch.normal(mean=0, std=self.std, size=(self.layers[l], self.layers[l + 1]),
                                             requires_grad=True)
        self.weights_shared = [weights_shared[i] for i in range(self.cross_layer)]

    def forward(self, user, item_d1, item_d2):
        user_emb = self.U(torch.LongTensor(user))
        item_emb_d1 = self.V_d1(torch.LongTensor(item_d1))
        item_emb_d2 = self.V_d2(torch.LongTensor(item_d2))

        cur_d1 = torch.cat((user_emb, item_emb_d1), 1)
        cur_d2 = torch.cat((user_emb, item_emb_d2), 1)
        pre_d1 = cur_d1
        pre_d2 = cur_d2
        for l in range(len(self.layers) - 1):
            cur_d1 = torch.add(torch.matmul(cur_d1, self.weights_d1[l]), self.biases_d1[l])
            cur_d2 = torch.add(torch.matmul(cur_d2, self.weights_d2[l]), self.biases_d2[l])

            if (l < self.cross_layer):
                cur_d1 = torch.matmul(pre_d2, self.weights_shared[l])
                cur_d2 = torch.matmul(pre_d1, self.weights_shared[l])

            cur_d1 = nn.functional.relu(cur_d1)
            cur_d2 = nn.functional.relu(cur_d2)
            pre_d1 = cur_d1
            pre_d2 = cur_d2

        z_d1 = torch.matmul(cur_d1, self.W_d1) + self.b_d1
        z_d2 = torch.matmul(cur_d2, self.W_d2) + self.b_d2

        return self.sigmoid(z_d1), self.sigmoid(z_d2)

    def fit(self, data, labels, num_epoch=101):

        params = [{"params": self.parameters(), "lr": self.lr},
                  {"params": self.weights_shared, "lr": self.lr, "weight_decay": self.reg}]
        optimizer = torch.optim.Adam(params)
        criterion = nn.MSELoss()

        train_data = torch.tensor(data, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        labels_d1, labels_d2 = labels[:, 0], labels[:, 1]
        user, item_d1, item_d2 = train_data[:, 0], train_data[:, 1], train_data[:, 2]
        for epoch in range(num_epoch):
            permutation = torch.randperm(user.shape[0])
            max_idx = int((len(permutation) // (self.batch_size / 2) - 1) * (self.batch_size / 2))
            for batch in range(0, max_idx, self.batch_size):
                optimizer.zero_grad()
                idx = permutation[batch: batch + self.batch_size]
                pred_d1, pred_d2 = self.forward(user[idx], item_d1[idx], item_d2[idx])
                loss_d1 = criterion(labels_d1[idx].float(), torch.squeeze(pred_d1))
                loss_d2 = criterion(labels_d2[idx].float(), torch.squeeze(pred_d2))
                loss = loss_d1 + loss_d2
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print("epoch {} loss: {:.4f}".format(epoch, loss))

