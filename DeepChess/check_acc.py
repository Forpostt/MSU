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

def check_acc():
    if GPU_USE:
        torch.cuda.set_device(DEVICE_NUM)

    win_data_test = Boards('./data/win_games.txt', max_size=50000)
    win_data_test.read_games()
    win_dataloader_test = DataLoader(win_data_test, batch_size=BATCH_SIZE, shuffle=True)
    lose_data_test = Boards('./data/lose_games.txt', max_size=50000)
    lose_data_test.read_games()
    lose_dataloader_test = DataLoader(lose_data_test, batch_size=BATCH_SIZE, shuffle=True)

    pretrained = torch.load('./data/pos2vec.pth.tar')
    model = torch.load('./data/model_2.pth.tar')
    
    correct = 0
    count = 0
    it = iter(lose_dataloader_test)
    for i, sample in enumerate(win_dataloader_test):
        X, y = shuffle_tenzor(sample.type(DTYPE), it.next().type(DTYPE), pretrained)

        prediction = model.forward(Variable(X)).data
        count += sample.shape[0]
        correct += (np.argmax(prediction.cpu().numpy(), 1) == y.cpu().numpy()).sum()
    print(float(correct) / float(count))

    
if __name__ == '__main__':
    GPU_USE = True
    DEVICE_NUM = 2
    DTYPE = torch.cuda.FloatTensor
    DTYPE_2 = torch.cuda.LongTensor
    check_acc()