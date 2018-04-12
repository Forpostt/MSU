import torch
from torch.utils.data import Dataset, DataLoader
import random

VEC_SIZE = 769
MAX_SIZE_DATASET = 250000


class Boards(Dataset):
    def __init__(self, txt_file_1=None, txt_file_2=None, max_size=MAX_SIZE_DATASET, type=torch.FloatTensor):
        self.boards = []
        self.txt_file_1 = txt_file_1
        self.txt_file_2 = txt_file_2
        self.max_size = max_size
        self.type = type

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = torch.zeros(VEC_SIZE).type(self.type)
        for i in self.boards[idx]:
            board[i] = 1

        return board

    def read_games(self):
        if self.txt_file_1 is not None:
            with open(self.txt_file_1, 'r') as f:
                lines = f.read().splitlines()
                i = 0
                while i < self.max_size:
                    line = random.choice(lines)
                    line = map(int, line.strip().split())
                    self.boards.append(line)
                    i += 1

        if self.txt_file_2 is not None:
            with open(self.txt_file_2, 'r') as f:
                lines = f.read().splitlines()
                i = 0
                while i < self.max_size:
                    line = random.choice(lines)
                    line = map(int, line.strip().split())
                    self.boards.append(line)
                    i += 1
