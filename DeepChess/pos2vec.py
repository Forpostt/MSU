import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import sys
import numpy as np
import pickle

BATCH_SIZE = 128
MAX_SIZE_DATASET = 250000    # 250000
VEC_SIZE = 769 
MAX_EPOCH = 30  # 40
DTYPE = torch.FloatTensor


class Boards(Dataset):
    def __init__(self, txt_file_1=None, txt_file_2=None):
        self.boards = []
        self.txt_file_1 = txt_file_1
        self.txt_file_2 = txt_file_2

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = torch.zeros(VEC_SIZE)
        for i in self.boards[idx]:
            board[i] = 1

        return board

    def read_games(self):
        if self.txt_file_1 is not None:
            with open(self.txt_file_1, 'r') as f:
                for i, line in enumerate(f):
                    if i >= MAX_SIZE_DATASET:
                        break
                    line = map(int, line.strip().split())
                    self.boards.append(line)

        if self.txt_file_2 is not None:
            with open(self.txt_file_2, 'r') as f:
                for i, line in enumerate(f):
                    if i >= MAX_SIZE_DATASET:
                        break
                    line = map(int, line.strip().split())
                    self.boards.append(line)

                    
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(autoencoder, pos2vec, dataloader, loss=torch.nn.MSELoss(), optim=torch.optim.Adam, max_epoch=MAX_EPOCH):
    train_loss_epochs = []
    optimizer = optim(autoencoder.parameters(), lr=0.005)
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
        #plt.figure(figsize=(8, 3))
        #plt.plot(train_loss_epochs)
        #plt.xlabel('Epoch', fontsize=16)
        #plt.ylabel('MSE', fontsize=16)
    except KeyboardInterrupt:
        pass
    


def Pos2Vec(layers=None):
    assert layers is not None and len(layers) > 1
    
    data = Boards('./data/win_games.txt', './data/lose_games.txt')
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

        optim = torch.optim.Adam
        loss = torch.nn.MSELoss()
        loss_step = train(autoencoder, pos2vec, dataloader, loss, optim, max_epoch=MAX_EPOCH)
        losses.append(loss_step)
        
        save_checkpoint(autoencoder, './data/autoen_{}_{}_{}.pth.tar'.format(layers[i], layers[i + 1], layers[i]))
            
    fc = list(pos2vec.children()) + [list(autoencoder.children())[1]]
    pos2vec = nn.Sequential(*fc)
    for param in pos2vec.parameters():
        param.requires_grad = False
    save_checkpoint(pos2vec, './data/pos2vec.pth.tar')
    
    pickle.dump(losses, open("./data/losses.p", "wb"))


if __name__ == '__main__':
    Pos2Vec(layers=[769, 500, 300, 100])
