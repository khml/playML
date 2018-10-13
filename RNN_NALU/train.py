# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_dataset(path='data.torch', split_position=3):
    data = torch.load(path)
    train_input = torch.from_numpy(data[split_position:, :-1])
    train_target = torch.from_numpy(data[split_position:, 1:])
    test_input = torch.from_numpy(data[:split_position, :-1])
    test_target = torch.from_numpy(data[:split_position, 1:])
    return train_input, train_target, test_input, test_target


class NAC(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.W_hat = Parameter(torch.Tensor(output_size, input_size))
        self.M_hat = Parameter(torch.Tensor(output_size, input_size))
        self.W = Parameter(torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat))

    def forward(self, input):
        return torch.matmul(input, self.W)


class NALU(nn.Module):
    def __init__(self, input_size, output_size, epsiron=1e-5):
        super().__init__()
        self.G = Parameter(torch.Tensor(output_size, input_size))
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.nac = NAC(output_size, input_size)
        self.epsiron = epsiron

    def forward(self, input):
        nac = self.nac(input)
        G = torch.sigmoid(torch.matmul(input, self.G.t()))
        log_in = torch.log(abs(input) + self.epsiron)
        m = torch.exp(torch.matmul(log_in, self.W.t()))
        return G * nac + (1 - G) * m


class RNNwithNALU(nn.Module):
    def __init__(self):
        super(RNNwithNALU, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.nalu = NALU(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.nalu(h_t2)
            outputs += [output]

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.nalu(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def main():
    split_position = 3
    train_input, train_target, test_input, test_target = load_dataset(split_position=split_position)
    rnn = RNNwithNALU()
    rnn.double()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(rnn.parameters(), lr=0.9)

    def train_model():
        def closure():
            optimizer.zero_grad()
            out = rnn(train_input)
            loss = criterion(out, train_target)
            print('train loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

    def eval_model(future=1000):
        with torch.no_grad():
            pred = rnn(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())

            x = test_input.detach().numpy()
            y = pred.detach().numpy()

            def plot_graph(x, y):
                plt.figure(figsize=(25, 5))
                plt.plot(np.arange(x.shape[0]), x, 'r:')
                plt.plot(np.arange(y.shape[0]), y, 'b:')
                plt.show()
                plt.close()

            for i in range(split_position):
                plot_graph(x[i], y[i])

    # begin to train
    for i in range(15):
        print('Epoch: ', i)
        train_model()
        eval_model()


if __name__ == '__main__':
    main()
