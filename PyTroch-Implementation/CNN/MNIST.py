# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms


def calc_length(origin_size, padding_size, dilation, kernel_size, stride):
    return (origin_size + 2 * padding_size - dilation * (kernel_size - 1) - 1) / stride + 1


def get_mnist_data_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # (1, 28, 28)  -> (32, 26, 26)
        self.conv2 = nn.Conv2d(32, 64, 3)  # (32, 26, 26) -> (64, 24, 24)
        self.pool = nn.MaxPool2d(2, 2)  # (64, 24, 24) -> (64, 12, 12)
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def main():
    epoch_num = 10
    model = CNN()

    train_loader, test_loader = get_mnist_data_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True)

    def train_model():
        model.train()
        running_loss = 0
        for (inputs, labels) in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print("loss = {}".format(loss.item()))
        print("epoch total loss = {}".format(running_loss))

    def eval_model():
        model.eval()
        with torch.no_grad():
            error = 0
            for (inputs, labels) in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                error += loss.item()
            print("loss = {}".format(error))

    for epoch in range(epoch_num):
        print("Epoch : {}".format(epoch))
        train_model()
        eval_model()


if __name__ == '__main__':
    main()
