import logging

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy
from tqdm import trange

logging.basicConfig(format="%(message)s", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 10)
        self.lr = 0.01
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


model = Net()

# Send the model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train = datasets.MNIST('', train=True, download=True, transform=transform)
    valid = datasets.MNIST('', train=False, download=True, transform=transform)

    trainloader = DataLoader(train, batch_size=32, shuffle=True)
    validloader = DataLoader(valid, batch_size=32, shuffle=True)

    min_lr = 1e-5
    max_lr = 10
    nb_epoch = 100
    gamma = np.geomspace(min_lr, max_lr, num=nb_epoch)

    optimizer = SGD(model.parameters(), lr=min_lr)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        mode='triangular2',
        base_lr=0.005/4,
        max_lr=0.005,
        step_size_up=10)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    # optimizer,
    # T_0=20,
    # T_mult=2)
    loss = nn.CrossEntropyLoss()

    data_plot = {'lr': [],
            'valid_loss': [],
            'train_loss': [],
            'accuracy': []}

    accuracy = MulticlassAccuracy(num_classes=10)
    accuracy.to('cuda')

    epoch = nb_epoch
    for e in trange(epoch):
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = gamma[e]

        train_loss, valid_loss = 0.0, 0.0

        # # Set model to training mode
        # model.train()
        # for data, label in trainloader:
        #     if torch.cuda.is_available():
        #         data, label = data.cuda(), label.cuda()
        #
        #     optimizer.zero_grad()
        #     target = model(data)
        #     train_step_loss = loss(target, label)
        #     train_step_loss.backward()
        #     optimizer.step()
        #
        #     train_loss += train_step_loss.item() * data.size(0)
        #
        # # Set model to Evaluation mode
        # model.eval()
        # for data, label in validloader:
        #     if torch.cuda.is_available():
        #         data, label = data.cuda(), label.cuda()
        #
        #     target = model(data)
        #     valid_step_loss = loss(target, label)
        #
        #     accuracy.update(target, label)
        #
        #     valid_loss += valid_step_loss.item() * data.size(0)
        # #
        # # Multiply the learning rate by 10 after each epoch
        scheduler.step()



        curr_lr = optimizer.param_groups[0]['lr']
        print(curr_lr)
        #
        # print(f'Epoch {e}\t \
        #         Training Loss: {train_loss / len(trainloader)}\t \
        #         Validation Loss:{valid_loss / len(validloader)}\t \
        #         LR:{curr_lr}')
        #
        # data_plot['train_loss'].append(train_loss / len(trainloader))
        # data_plot['valid_loss'].append(valid_loss / len(validloader))
        data_plot['lr'].append(curr_lr)
        # data_plot['e'].append(curr_lr)
        # data_plot['accuracy'].append(float(accuracy.compute().cpu().numpy()))

    plt.plot(list(range(nb_epoch)), data_plot['lr'])
    plt.show()



    # plt.plot(data_plot['lr'], data_plot['train_loss'], label='train_loss')
    # plt.plot(data_plot['lr'], data_plot['valid_loss'], label='valid_loss')
    # plt.legend()
    # plt.show()
    #
    # plt.semilogx(data_plot['lr'], data_plot['accuracy'], label='accuracy')
    # plt.legend()
    # plt.show()
