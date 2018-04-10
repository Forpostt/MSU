import torch


def from_cuda_to_cpu(path='./data/pos2vec.pth.tar'):
    torch.cuda.set_device(2)
    model = torch.load(path)
    model = model.cpu()
    torch.save(model, './data/pos2vec_cpu.pth.tar')


if __name__ == '__main__':
    from_cuda_to_cpu()
