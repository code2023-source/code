import pandas as pd
import torch as th
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
from torch.utils.data import DataLoader
from scipy import linalg as sl
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

global i_txt
# i_txt = 'd2_concat'


def _FirstOderLR(x):
    # 转换为matrix
    x = np.mat(x)
    # 矩阵列为变量，行为实例
    x = x.T
    # print(cov, "\n")
    # 计算矩阵行和列
    N = x.shape[0]
    T = x.shape[1]
    # 标准化
    temp1 = np.mat(np.ones((1, T)))
    temp2 = (np.mean(x.T, axis=0)).T
    X = x - temp2 * temp1
    # print(X.shape[0], X.shape[1], "\n")
    eps = 0.0000000000000002220446049250313
    temp3 = (np.std(X.T, axis=0, ddof=1) + eps).T * temp1
    X = np.divide(X, temp3)
    # 计算协方差矩阵
    Cov = np.cov(X)
    temp4 = (X * np.tanh(X.T) - np.tanh(X) * X.T)
    # 计算likelihood radio
    LR = np.multiply(Cov, temp4) / T
    print(LR, "\n")
    return LR


def _h(W):
    d = 5
    M = np.eye(d) + W * W / d  # (Yu et al. 2019)
    E = np.linalg.matrix_power(M, d - 1)
    h = (E.T * M).sum() - d
    G_h = E.T * W * 2
    # print(G_h)
    return h, G_h


class RNN(nn.Module):
    def __init__(self, inputsize, hiddensize, outsize):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=inputsize,
            hidden_size=hiddensize,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(hiddensize, outsize)

    def forward(self, x):
        r_out, h_state = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


class CNN(nn.Module):
    def __init__(self, inputsize):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv1d(
                in_channels=inputsize,  # input height
                out_channels=16,  # n_filters
                kernel_size=101,  # filter size
                stride=1,  # filter movement/step
                padding=50,
            ),
            nn.ReLU(),  # activation
            nn.MaxPoold(kernel_size=2),
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Convd(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, sizes, **kwargs):
        super(Discriminator, self).__init__()
        self.sht = kwargs.get('shortcut', False)
        # activation_function = kwargs.get('activation_function', nn.ReLU)
        activation_function = kwargs.get('activation_function', nn.Sigmoid)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        dropout = kwargs.get("dropout", 0.)

        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, j))
            if batch_norm:
                layers.append(nn.BatchNorm1d(j))
            if dropout != 0.:
                layers.append(nn.Dropout(p=dropout))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)
        print(self.layers)

    def forward(self, x):
        return self.layers(x)


class FilterGenerator(nn.Module):  #效应连接生成器

    def __init__(self, sizes, zero_components=[], **kwargs):
        """Initialize a generator."""
        super(FilterGenerator, self).__init__()
        activation_function = kwargs.get('activation_function', nn.Tanh)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        layers = []

        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, j))
            # layers.append(nn.LSTM(input_size=i, hidden_size=j))
            # layers.append(RNN(i, j))
            if batch_norm:
                layers.append(nn.BatchNorm1d(j))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))
                layers.append(activation_function(activation_argument))

        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)

        # Filtering the unconnected nodes.
        self._filter = th.ones(1, sizes[0])  #因果参数
        for i in zero_components:
            self._filter[:, i].zero_()

        self._filter = Variable(self._filter, requires_grad=False)
        self.fs_filter = nn.Parameter(self._filter.data)

    def forward(self, x):
        return self.layers(x * (self._filter * self.fs_filter).expand_as(x))


class Generators(nn.Module):
    """Ensemble of all the generators."""

    def __init__(self, data_shape, zero_components, nh=200, batch_size=-1, **kwargs):
        """Init the model."""
        super(Generators, self).__init__()
        if batch_size == -1:
            batch_size = data_shape[0]
        rows, self.cols = data_shape

        # building the computation graph
        self.noise = [Variable(th.FloatTensor(batch_size, 1))
                      for i in range(self.cols)]
        self.blocks = th.nn.ModuleList()
        # Init all the blocks
        for i in range(self.cols):
            self.blocks.append(FilterGenerator(
                [self.cols + 1, nh, 1], zero_components[i], **kwargs))

    def forward(self, x):
        """Feed-forward the model."""
        for i in self.noise:
            i.data.normal_()

        self.generated_variables = [self.blocks[i](
            th.cat([x, self.noise[i]], 1)) for i in range(self.cols)]
        return self.generated_variables

def run_GANEC(df_data, skeleton=None, **kwargs):
    train_epochs = kwargs.get('train_epochs', 1000)
    test_epochs = kwargs.get('test_epochs', 1000)
    batch_size = kwargs.get('batch_size', -1)

    lr_gen = kwargs.get('lr_gen', 0.1)
    lr_disc = kwargs.get('lr_disc', lr_gen)
    verbose = kwargs.get('verbose', True)
    regul_param = kwargs.get('regul_param', 0.1)
    rou = kwargs.get('rou', 1)
    maxp = kwargs.get('rou', 2)
    dnh = kwargs.get('dnh', None)

    d_str = "Epoch: {} -- Disc: {} -- Gen: {} -- L1: {}"
    try:
        list_nodes = list(df_data.columns)
        df_data = (df_data[list_nodes]).as_matrix()
    except AttributeError:
        list_nodes = list(range(df_data.shape[1]))
    data = df_data.astype('float32')
    data = th.from_numpy(data)
    if batch_size == -1:
        batch_size = data.shape[0]
    rows, cols = data.size()

    if skeleton is not None:
        zero_components = [[] for i in range(cols)]
        for i, j in zip(*((1 - skeleton).nonzero())):
            zero_components[j].append(i)
    else:
        zero_components = [[i] for i in range(cols)]
    sam = Generators((rows, cols), zero_components, batch_norm=True, **kwargs)

    activation_function = kwargs.get('activation_function', th.nn.Tanh)
    # LR = _FirstOderLR(data)
    # print(LR)

    try:
        del kwargs["activation_function"]
    except KeyError:
        pass
    discriminator_sam = Discriminator(
        [cols, dnh, dnh, 1], batch_norm=True,
        activation_function=th.nn.Sigmoid,
        activation_argument=None, **kwargs)
    # discriminator_sam = Discriminator(
    #     [cols, dnh, dnh, 1], batch_norm=True,
    #     activation_function=th.nn.LeakyReLU,
    #     activation_argument=0.2, **kwargs)
    kwargs["activation_function"] = activation_function

    # Select parameters to optimize : ignore the non connected nodes
    criterion = th.nn.BCEWithLogitsLoss()
    g_optimizer = th.optim.Adam(sam.parameters(), lr=lr_gen)
    d_optimizer = th.optim.Adam(discriminator_sam.parameters(), lr=lr_disc)

    true_variable = Variable(th.ones(batch_size, 1), requires_grad=False)
    false_variable = Variable(th.zeros(batch_size, 1), requires_grad=False)
    causal_filters = th.zeros(data.shape[1], data.shape[1])

    data_iterator = DataLoader(data, batch_size=batch_size, shuffle=True)

    # TRAIN
    for epoch in range(train_epochs + test_epochs):
        for i_batch, batch in enumerate(data_iterator):
            batch = Variable(batch)
            batch_vectors = [batch[:, [i]] for i in range(cols)]

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # Train the discriminator
            generated_variables = sam(batch)
            disc_losses = []
            gen_losses = []

            for i in range(cols):
                generator_output = th.cat([v for c in [batch_vectors[: i], [
                    generated_variables[i]],
                                                       batch_vectors[i + 1:]] for v in c], 1)
                # 1. Train discriminator on fake
                disc_output_detached = discriminator_sam(
                    generator_output.detach())
                disc_output = discriminator_sam(generator_output)
                disc_losses.append(
                    criterion(disc_output_detached, false_variable))

                # 2. Train the generator :
                gen_losses.append(criterion(disc_output, true_variable))

            true_output = discriminator_sam(batch)
            adv_loss = sum(disc_losses) / cols + \
                       criterion(true_output, true_variable)
            gen_loss = sum(gen_losses)

            adv_loss.backward()
            d_optimizer.step()

            # 3. Compute filter regularization
            filters = th.stack([i.fs_filter[0, :-1].abs() for i in sam.blocks], 1)

            # 损失函数设计L1正则
            l1_reg = 0.5 * regul_param * filters.sum() * math.log(200)
            # 损失函数加DAG约束
            A = filters.data
            W = A.numpy()
            h, G_h = _h(W)
            dag = 0.5 * rou * (h * h)
            # dag = 0.5 * rou * h * h
            # dag = th.from_numpy(dag)
            # lr=_FirstOderLR(data)
            # lr=np.array(lr)
            # lr = th.from_numpy(lr)
            # lr[lr < 0] = 0
            # 损失函数
            loss = gen_loss + l1_reg + dag

            if verbose and epoch % 200 == 0 and i_batch == 0:
                string_txt = str(i) + " " + d_str.format(epoch,
                                                         adv_loss.cpu().item(),
                                                         gen_loss.cpu(
                                                         ).item() / cols,
                                                         l1_reg.cpu().item())
                print(string_txt)
                print(causal_filters)
                print(type(causal_filters))
                global i_txt
                if epoch == 300:
                    np.savetxt('D:/ECGAN/result.txt', np.array(causal_filters), fmt="%f",
                               delimiter=",")
            loss.backward()

            if epoch > train_epochs:
                causal_filters.add_(filters.data)
            g_optimizer.step()

    tp = causal_filters.div_(10).cpu().numpy()
    # for i in range(tp.shape[0]):
    #     for j in range(causal_filters.shape[1]):
    #         if causal_filters.div_(10).numpy()[i][j]>causal_filters.div_(10)numpy()[j][i]:

    return causal_filters.div_(10).cpu().numpy()

class GANEC(object):
    def __init__(self, lr=0.1, dlr=0.1,l1=0.1, nh=200, dnh=200,
                 train_epochs=1000, test_epochs=1000, batchsize=-1):
        super(GANEC, self).__init__()
        self.lr = lr
        self.dlr = dlr #调参
        self.l1 = l1   #调参
        self.nh = nh
        self.dnh = dnh
        self.train = train_epochs
        self.test = test_epochs
        self.batchsize = batchsize

    def Train_RUN(self, data, graph=None):
        list_out = [run_GANEC(data, skeleton=graph,
                              lr_gen=self.lr, lr_disc=self.dlr,
                              regul_param=self.l1, nh=self.nh, dnh=self.dnh,
                              train_epochs=self.train,
                              test_epochs=self.test, batch_size=self.batchsize)]

        return list_out

    def Test_RUN(self, data):
        list_out = [run_GANEC(data, skeleton=None,
                              lr_gen=self.lr, lr_disc=self.dlr,
                              regul_param=self.l1, nh=self.nh, dnh=self.dnh,
                              train_epochs=self.train,
                              test_epochs=self.test, batch_size=self.batchsize)]
        print(list_out)
        np.savetxt('D:/ECGAN/result/result1.txt', list_out[0], fmt="%f", delimiter=",")
        return list_out


if __name__ == '__main__':
    GAN = GANEC(lr=0.01, dlr=0.01, l1=5, nh=100, dnh=100, train_epochs=400, test_epochs=1600)  #5各结点
    p_txt = r"D:/ECGAN/result/result2.txt"
    GAN1.Test_RUN(data=np.array(pd.read_csv(p_txt, sep=" "))[:,1:])
