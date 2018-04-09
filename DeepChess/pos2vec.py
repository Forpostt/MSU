import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import sys
import numpy as np
import pickle
from boards import Boards

BATCH_SIZE = 256  # 250000
VEC_SIZE = 769
MAX_EPOCH = 30  # 40
GPU_USE = False
DEVICE_NUM = 0
DTYPE = torch.FloatTensor


def train(autoencoder, pos2vec, dataloader, loss, optim, max_epoch=MAX_EPOCH):
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

            train_loss_epochs.append(np.mean(losses))
            sys.stdout.write('\rEpoch {0}... Train MSE: {1:.6f}'.format(epoch, train_loss_epochs[-1]))
        
        return train_loss_epochs
    except KeyboardInterrupt:
        pass


def Pos2Vec(layers=None):
    if layers is None:
        layers = [769, 500, 300, 100]
    assert len(layers) > 1
    
    data = Boards('./data/win_games.txt', './data/lose_games.txt', type=DTYPE)
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

        autoencoder = nn.Sequential(nn.BatchNorm1d(layers[i]), nn.Linear(layers[i], layers[i + 1]), 
                                    nn.BatchNorm1d(layers[i + 1]), nn.ReLU(), 
                                    nn.Linear(layers[i + 1], layers[i]))
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
    Pos2Vec(layers=[769, 500, 300, 100])
