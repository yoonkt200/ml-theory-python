# Logistic Regression Classifier : It is only for study
# Copyright : yoonkt200@gmail.com
# Apache License 2.0
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score


class LibsvmTransformer:
    def __init__(self, train_path, valid_path, batch_size, input_dim):
        self._train_path = train_path
        self._valid_path = valid_path
        self._batch_size = batch_size
        self._input_dim = input_dim
        self.init_stream()

    def init_stream(self):
        """
        Init data load stream
        """
        self._train_streamer = self.stream_docs(path=self._train_path)

    def stream_docs(self, path):
        """
        Lazy function (generator) to read a file piece by piece.
        Transform libsvm data format to numpy array.
        """
        with open(path, 'r', encoding='utf-8') as txt:
            for line in txt:
                y = int(line[0])
                x = np.zeros(self._input_dim)
                for tup in line[2:].split(" "):
                    x[int(tup.split(":")[0])] = int(tup.split(":")[1])
                yield x, y

    def get_minibatch(self):
        """
        Get mini-batch from dataset generator
        """
        X = []
        Y = []
        try:
            for _ in range(self._batch_size):
                x_i, y_i = next(self._train_streamer)
                X.append(x_i)
                Y.append(y_i)
        except StopIteration:
            return None, None
        return np.array(X, dtype=np.float32), np.array(Y)

    def get_valid_batch(self):
        """
        get batch valid data
        """
        valid_file = open(self._valid_path, 'r')
        valid_data = valid_file.read().split('\n')
        valid_file.close()
        X = []
        Y = []
        for line in valid_data:
            y = int(line[0])
            x = np.zeros(self._input_dim)
            for tup in line[2:].split(" "):
                x[int(tup.split(":")[0])] = int(tup.split(":")[1])
            X.append(x)
            Y.append(y)
        return np.array(X, dtype=np.float32), np.array(Y)


class WideAndDeep(torch.nn.Module):
    """
    Define Logistic Regression Model
    """
    def __init__(self, input_dim, num_class):
        super(WideAndDeep, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.linear = torch.nn.Linear(input_dim + 8, num_class)

    def forward(self, wide_input, deep_input):
        deep = self.fc1(deep_input)
        deep = self.relu(deep)
        deep = self.fc2(deep)
        wide_n_deep = torch.cat((wide_input, deep), dim=1)
        y_pred = self.linear(wide_n_deep)
        return y_pred


def check_early_stop(check_list, early_stop_window):
    """
    Check early stop
    """
    if check_list[-1] > check_list[-early_stop_window]:
        return True
    else:
        return False


def train(input_dim, num_class, epoch, early_stop_window):
    # model, dataset define
    data_transformer = LibsvmTransformer(train_path='../dataset/train.txt',
                                         valid_path='../dataset/test.txt',
                                         input_dim=input_dim,
                                         batch_size=128)
    model = WideAndDeep(input_dim, num_class)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    x_valid, y_valid = data_transformer.get_valid_batch()
    valid_inputs = torch.from_numpy(x_valid)
    valid_targets = torch.from_numpy(y_valid)

    # iterate epoch
    iter_epoch = 0
    valid_loss_list = []
    while iter_epoch < epoch:
        x_train, y_train = data_transformer.get_minibatch()
        if x_train is not None:
            inputs = torch.from_numpy(x_train)
            targets = torch.from_numpy(y_train)

            # forward pass
            outputs = model(wide_input=inputs, deep_input=inputs)
            loss = criterion(outputs, targets)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_auc = roc_auc_score(targets.detach().numpy(), outputs.detach().numpy()[:, 1])

            # validation
            valid_outputs = model(wide_input=valid_inputs, deep_input=valid_inputs)
            valid_loss = criterion(valid_outputs, valid_targets)
            valid_auc = roc_auc_score(valid_targets.detach().numpy(), valid_outputs.detach().numpy()[:, 1])
        else:
            valid_loss_list.append(valid_loss.item())
            if iter_epoch > 5:
                if check_early_stop(check_list=valid_loss_list, early_stop_window=early_stop_window):
                    print('Epoch [{}/{}], Loss: {:.4f}, Valid Loss: {:.4f}, AUC: {:.4f}, Valid AUC: {:.4f},'.format(iter_epoch, epoch, loss.item(), valid_loss.item(), train_auc, valid_auc))
                    print("Early stop at:", iter_epoch)
                    return
            print('Epoch [{}/{}], Loss: {:.4f}, Valid Loss: {:.4f}, AUC: {:.4f}, Valid AUC: {:.4f},'.format(iter_epoch, epoch, loss.item(), valid_loss.item(), train_auc, valid_auc))
            data_transformer.init_stream()
            iter_epoch += 1


if __name__ == "__main__":
    train(input_dim=37, num_class=2, epoch=2000, early_stop_window=3)
