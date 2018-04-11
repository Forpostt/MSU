import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from boards import Boards
import numpy as np
import pickle
import sys
import random


DTYPE = torch.FloatTensor
BATCH_SIZE = 64
MAX_EPOCHS = 20
VEC_SIZE = 100
GPU_USE = False
DEVICE_NUM = 0
LEARNING_RATE = 0.01
MAX_DATASET_SIZE = 10000


def shuffle_tenzor(good_sample, bad_sample, pretrained):
    X = torch.zeros(good_sample.shape[0], 2 * VEC_SIZE).type(DTYPE)
    y = torch.zeros(good_sample.shape[0], 2).type(DTYPE)

    sample_0 = pretrained.forward(Variable(good_sample)).data
    sample_1 = pretrained.forward(Variable(bad_sample)).data
    for i in range(good_sample.shape[0]):
        if random.randint(0, 1):
            X[i, :VEC_SIZE] = sample_0[i]
            X[i, VEC_SIZE:] = sample_1[i]
            y[i, 0] = 1
        else:
            X[i, :VEC_SIZE] = sample_1[i]
            X[i, VEC_SIZE:] = sample_0[i]
            y[i, 1] = 1
    return X, y


def train(model, pretrained, win_dataloader, lose_dataloader, loss, optim, max_epochs=MAX_EPOCHS):
    train_loss_epochs = []
    train_acc = []
    optimizer = optim(model.parameters(), lr=LEARNING_RATE)
    try:
        for epoch in range(max_epochs):
            losses = []
            it = iter(lose_dataloader)
            for i, sample in enumerate(win_dataloader):
                X, y = shuffle_tenzor(sample, it.next(), pretrained)

                optimizer.zero_grad()
                prediction = model.forward(Variable(X))

                loss_batch = loss(prediction, Variable(y))
                losses.append(loss_batch.data[0])

                loss_batch.backward()
                optimizer.step()

            train_loss_epochs.append(np.mean(losses))

            right_answer = 0
            data_size = 0
            it = iter(lose_dataloader)
            for i, sample in enumerate(win_dataloader):
                X, y = shuffle_tenzor(sample, it.next(), pretrained)

                prediction = model.forward(Variable(X)).data
                data_size += sample.shape[0]
                right_answer += ((prediction * y).sum(1) > 0.5).sum()

            train_acc.append(float(right_answer) / float(data_size))

            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.99
            
            if epoch % 5 == 4:
                torch.save(model, './data/model_2.pth.tar')
 
            print('\rEpoch {0}... MSE: {1:.6f}  Acc: {2:.6}'.format(epoch, train_loss_epochs[-1], train_acc[-1]))
        return train_loss_epochs, train_acc

    except KeyboardInterrupt:
	print('KeyboardInterrup!!!')
        return train_loss_epochs, train_acc


def DeepChess(layers=None):
    if layers is None:
        layers = [200, 400, 200, 100, 2]
    assert len(layers) > 1

    win_data = Boards('./data/win_games.txt', max_size=MAX_DATASET_SIZE, type=DTYPE)
    win_data.read_games()
    win_dataloader = DataLoader(win_data, batch_size=BATCH_SIZE, shuffle=True)
    lose_data = Boards('./data/lose_games.txt', max_size=MAX_DATASET_SIZE, type=DTYPE)
    lose_data.read_games()
    lose_dataloader = DataLoader(lose_data, batch_size=BATCH_SIZE, shuffle=True)

    pretrained = torch.load('./data/pos2vec_cpu.pth.tar')

    fc = [nn.BatchNorm1d(layers[0])]
    for i in range(len(layers))[:-2]:
        fc += [nn.Linear(layers[i], layers[i + 1]), nn.BatchNorm1d(layers[i + 1]), nn.ReLU()]
    fc += [nn.Linear(layers[-2], layers[-1]), nn.Softmax(1)]
    model = nn.Sequential(*fc)

    if GPU_USE:
        torch.cuda.set_device(DEVICE_NUM)
        model = model.cuda()
        pretrained = pretrained.cuda()

    loss = nn.MSELoss()
    optim = torch.optim.RMSprop

    losses, acc = train(model, pretrained, win_dataloader, lose_dataloader, loss, optim, MAX_EPOCHS)
    pickle.dump([losses, acc], open('./data/model_acc_loss_2.p', 'w'))
    torch.save(model, './data/model_2.pth.tar')


if __name__ == '__main__':
    GPU_USE = True
    DTYPE = torch.cuda.FloatTensor
    BATCH_SIZE = 256
    MAX_EPOCHS = 50
    DEVICE_NUM = 2
    MAX_DATASET_SIZE = 500000
    DeepChess()
