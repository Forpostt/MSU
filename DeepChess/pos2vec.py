import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import sys
import numpy as np
import pickle
from boards import Boards
import matplotlib.pyplot as plt


BATCH_SIZE = 64  # 250000
VEC_SIZE = 769
MAX_SIZE = 5000
MAX_EPOCH = 10  # 40
GPU_USE = False
DEVICE_NUM = 0
LEARNING_RATE = 0.005
DTYPE = torch.FloatTensor


def init_he_linear(input_size, output_size):
    layer = nn.Linear(input_size, output_size)
    weight = layer.state_dict()['weight']
    bias = layer.state_dict()['bias']
    bias.zero_()
    weight.normal_(0, np.sqrt(float(2) / float(output_size)))
    return layer


def train(autoencoder, pos2vec, dataloader, loss, optim, max_epoch=MAX_EPOCH, eps=1e-5):
    train_loss_epochs = []
    optimizer = optim(autoencoder.parameters(), lr=0.001)
    try:
        for epoch in range(max_epoch):
            losses = []
            for sample in dataloader:
                X = Variable(sample.type(DTYPE))
                y = pos2vec.forward(X)

                optimizer.zero_grad()

                prediction = autoencoder.forward(y)
                loss_batch = loss(prediction, y)
                losses.append(loss_batch.data[0])

                loss_batch.backward()
                optimizer.step()

            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.98

            train_loss_epochs.append(np.mean(losses))

            if len(train_loss_epochs) > 1 and abs(train_loss_epochs[-1] - train_loss_epochs[-2]) < eps:
                print "Delta of error is smaller then eps!"
                return train_loss_epochs

            print('\rEpoch {0}... Train MSE: {1:.6f}'.format(epoch, train_loss_epochs[-1]))

        return train_loss_epochs
    except KeyboardInterrupt:
        print 'KeyboardInterrupt'
        return train_loss_epochs


def Pos2Vec(layers=None):
    if layers is None:
        layers = [769, 600, 400, 200, 100]
    assert len(layers) > 1

    data = Boards('./data/win_games.txt', './data/lose_games.txt', max_size=MAX_SIZE)
    data.read_games()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    pos2vec = None
    autoencoder = None
    losses = []
    for i in range(len(layers))[:-1]:
        if pos2vec is None:
            pos2vec = nn.Sequential(nn.BatchNorm1d(layers[0]))
        else:
            fc = list(pos2vec.children()) + list(autoencoder.children())[1:-1]
            pos2vec = nn.Sequential(*fc)

        for param in pos2vec.parameters():
            param.requires_grad = False

        autoencoder = nn.Sequential(nn.BatchNorm1d(layers[i]), init_he_linear(layers[i], layers[i + 1]),
                                    nn.BatchNorm1d(layers[i + 1]), nn.ReLU(),
                                    init_he_linear(layers[i + 1], layers[i]))
        if GPU_USE:
            torch.cuda.set_device(DEVICE_NUM)
            autoencoder = autoencoder.cuda()
            pos2vec = pos2vec.cuda()

        optim = torch.optim.RMSprop
        loss = torch.nn.MSELoss()
        loss_step = train(autoencoder, pos2vec, dataloader, loss, optim, max_epoch=MAX_EPOCH)
        losses.append(loss_step)
            
    fc = list(pos2vec.children()) + [list(autoencoder.children())[1]]
    pos2vec = nn.Sequential(*fc)
    for param in pos2vec.parameters():
        param.requires_grad = False
    torch.save(pos2vec, './data/pos2vec.pth.tar')
    
    pickle.dump(losses, open("./data/losses.p", "wb"))


if __name__ == '__main__':
    DTYPE = torch.cuda.FloatTensor
    GPU_USE = True
    DEVICE_NUM = 2
    MAX_EPOCH = 50
    BATCH_SIZE = 256
    MAX_SIZE = 500000
    Pos2Vec(layers=[769, 600, 400, 200, 100])
