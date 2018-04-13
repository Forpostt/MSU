import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from boards import Boards
import numpy as np
import pickle
import sys
import random


DTYPE = torch.FloatTensor
DTYPE_2 = torch.LongTensor
BATCH_SIZE = 64
MAX_EPOCHS = 70
VEC_SIZE = 100
GPU_USE = False
DEVICE_NUM = 0
LEARNING_RATE = 0.01
MAX_DATASET_SIZE = 10000


def init_he_linear(input_size, output_size):
    layer = nn.Linear(input_size, output_size)
    weight = layer.state_dict()['weight']
    bias = layer.state_dict()['bias']
    bias.zero_()
    weight.normal_(0, np.sqrt(float(2) / float(output_size)))
    return layer


def shuffle_tenzor(good_sample, bad_sample, pretrained):
    X = torch.zeros(good_sample.shape[0], 2 * VEC_SIZE).type(DTYPE)
    y = torch.zeros(good_sample.shape[0]).type(DTYPE_2)

    sample_0 = pretrained.forward(Variable(good_sample)).data
    sample_1 = pretrained.forward(Variable(bad_sample)).data
    for i in range(good_sample.shape[0]):
        if random.randint(0, 1):
            X[i, :VEC_SIZE] = sample_0[i]
            X[i, VEC_SIZE:] = sample_1[i]
            y[i] = 0
        else:
            X[i, :VEC_SIZE] = sample_1[i]
            X[i, VEC_SIZE:] = sample_0[i]
            y[i] = 1
    return X, y


def train(model, pretrained, win_dataloader, lose_dataloader, win_dataloader_test, lose_dataloader_test, loss, optim, max_epochs=MAX_EPOCHS):
    train_loss_epochs = []
    train_acc = []
    optimizer = optim(model.parameters(), lr=LEARNING_RATE)
    try:
        for epoch in range(max_epochs):
            losses = []
            it = iter(lose_dataloader)
            for i, sample in enumerate(win_dataloader):
                X, y = shuffle_tenzor(sample.type(DTYPE), it.next().type(DTYPE), pretrained)

                optimizer.zero_grad()
                prediction = model.forward(Variable(X))

                loss_batch = loss(prediction, Variable(y))
                losses.append(loss_batch.data[0])

                loss_batch.backward()
                optimizer.step()

            train_loss_epochs.append(np.mean(losses))

            correct = 0
            count = 0
            it = iter(lose_dataloader_test)
            for i, sample in enumerate(win_dataloader_test):
                X, y = shuffle_tenzor(sample.type(DTYPE), it.next().type(DTYPE), pretrained)

                prediction = model.forward(Variable(X)).data
                count += sample.shape[0]
                correct += (np.argmax(prediction.cpu().numpy(), 1) == y.cpu().numpy()).sum()

            train_acc.append(float(correct) / float(count))

            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.99

            if epoch % 5 == 4:
                torch.save(model, './data/model_2.pth.tar')

            if len(train_acc) > 1 and abs(train_acc[-1] - train_acc[-2]) < 1e-5:
                print('Delta is smaller than eps')
                return train_loss_epochs, train_acc

            print('\rEpoch {0}... MSE: {1:.6f}  Acc: {2:.6}'.format(epoch,
                                                                               train_loss_epochs[-1],
                                                                               train_acc[-1]))
        return train_loss_epochs, train_acc

    except KeyboardInterrupt:
        print('KeyboardInterrupt!')
        return train_loss_epochs, train_acc


def DeepChess(layers=None):
    if layers is None:
        layers = [200, 400, 200, 100, 2]
    assert len(layers) > 1

    if GPU_USE:
        torch.cuda.set_device(DEVICE_NUM)

    win_data = Boards('./data/win_games.txt', max_size=MAX_DATASET_SIZE)
    win_data.read_games()
    win_dataloader = DataLoader(win_data, batch_size=BATCH_SIZE, shuffle=True)
    lose_data = Boards('./data/lose_games.txt', max_size=MAX_DATASET_SIZE)
    lose_data.read_games()
    lose_dataloader = DataLoader(lose_data, batch_size=BATCH_SIZE, shuffle=True)

    win_data_test = Boards('./data/win_games.txt', max_size=50000)
    win_data_test.read_games()
    win_dataloader_test = DataLoader(win_data_test, batch_size=BATCH_SIZE, shuffle=True)
    lose_data_test = Boards('./data/lose_games.txt', max_size=50000)
    lose_data_test.read_games()
    lose_dataloader_test = DataLoader(lose_data_test, batch_size=BATCH_SIZE, shuffle=True)

    pretrained = torch.load('./data/pos2vec.pth.tar')

    fc = [nn.BatchNorm1d(layers[0])]
    for i in range(len(layers))[:-2]:
        fc += [init_he_linear(layers[i], layers[i + 1]), nn.BatchNorm1d(layers[i + 1]), nn.ReLU()]
    fc += [init_he_linear(layers[-2], layers[-1]), nn.Softmax(1)]
    model = nn.Sequential(*fc)

    if GPU_USE:
        model = model.cuda()
        pretrained = pretrained.cuda()

    loss = nn.CrossEntropyLoss()
    optim = torch.optim.RMSprop

    losses, acc = train(model, pretrained, win_dataloader, lose_dataloader,
                        win_dataloader_test, lose_dataloader_test, loss, optim)
    pickle.dump([losses, acc], open('./data/model_acc_loss_2.p', 'w'))
    torch.save(model, './data/model_2.pth.tar')


if __name__ == '__main__':
    GPU_USE = True
    DTYPE = torch.cuda.FloatTensor
    DTYPE_2 = torch.cuda.LongTensor
    BATCH_SIZE = 256
    MAX_EPOCHS = 50
    MAX_DATASET_SIZE = 500000
    DEVICE_NUM = 2
    DeepChess()
