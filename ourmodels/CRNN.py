import random
from optparse import OptionParser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.collections import LineCollection
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        e1 = F.leaky_relu(self.conv0(x))
        # print(e1.shape)
        e2 = F.leaky_relu(self.conv1(e1))
        # print(e2.shape)
        e3 = self.conv2(e2)

        return e3


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 6

        # CNN
        self.conv0 = nn.Conv1d(
            in_channels=1, out_channels=self.input_size,
            kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(
            in_channels=self.input_size, out_channels=self.input_size,
            kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv1d(in_channels=6, out_channels=1, kernel_size=3, stride=1, padding=1)

        # LSTM

        # self.time_period = 30
        self.hidden_layer = 5
        self.directional = 2
        self.rnn = nn.LSTM(self.input_size, self.hidden_layer, 2, bidirectional=self.directional==2)
        # self.h0 = torch.randn(2, 3, 20)
        # self.c0 = torch.randn(2, 3, 20)

        self.fc = nn.Linear(self.hidden_layer*self.directional, 1)

    def forward(self, x):

        # CNN
        e1 = F.leaky_relu(self.conv0(x))
        # print(e1.shape)
        e2 = F.leaky_relu(self.conv1(e1))
        # print(e2.shape)
        # e3 = self.conv2(e2)

        # b, c, w = e2.size()  # batch, input_size, seq_len
        # assert h == 1, "the height of conv must be 1"
        # conv = conv.squeeze(2)
        # pad_to = (w // self.time_period + ((w % self.time_period) != 0)) * self.time_period
        # e2 = F.pad(e2, (0, pad_to - w), mode='constant', value=0)
        # b, c, w = e2.size()
        e2 = e2.permute(2, 0, 1)  # [w, b, c]  (seq_len, batch, input_size)

        # LSTM
        # outputs = []
        # for i in range(0, w, self.lstm_len):
        rnn_output, (hn, cn) = self.rnn(e2)  #, (self.h0, self.c0))

        rnn_output = rnn_output.squeeze(1)

        # print(rnn_output.shape)
        # outputs.append(output)

        # outputs = torch.cat(outputs)
        # outputs = outputs[:w_initial]

        output = self.fc(rnn_output)
        output = output.squeeze(1)
        # print(output.shape)

        return output


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # print(outputs.shape)
    # print(labels.shape)
    outputs = outputs.byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum()  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum()         # Will be zero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.mean(iou)
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  #


def get_color(y_pred, y_mb):
    #  зелёный/синий -- 0 правильный/неправильный
    #  красный/черный -- 1 правильный/неправильный
    # 1 -- правильный
    if y_pred:
        if y_mb:
            return "red"
        else:
            return "black"
    # 0 -- правильный
    else:
        if y_mb:
            return "blue"
        else:
            return "green"


def draw_pics(x, y, y_predicted, y_must_be):

    colors = [
        get_color(y_pred, y_mb)
        for y_pred, y_mb in zip(y_predicted, y_must_be)
    ]
    points = np.array([x, y]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lines = LineCollection(segments, colors=colors)

    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())

    plt.gca().add_collection(lines)

    plt.show()


def draw_test(x, y, y_predicted):

    colors = [
        "red" if y_pred else "green"
        for y_pred in y_predicted
    ]
    points = np.array([x, y]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lines = LineCollection(segments, colors=colors)

    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())

    plt.gca().add_collection(lines)

    plt.show()


def init_weights(m):
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.1) ## or simply use your layer.reset_parameters()
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(1 / m.in_features))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(4 / m.in_channels))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def cross_validation(
        model, opt, loss_fn, lr_decay, epochs, X, Y, device, pics_num=None):

    threshold = 0.5
    fold_ks = 3

    kf = KFold(n_splits=fold_ks, random_state=8, shuffle=True)

    all_train_ious = []
    all_test_ious = []
    all_train_loss = []
    all_test_loss = []

    fold_k = 0

    f1_scores = []

    for train_index, test_index in kf.split(X):

        if pics_num is not None:
            draw_for = random.sample(list(test_index), pics_num)
        else:
            draw_for = list(test_index)

        model.apply(init_weights)

        print('* Fold %d/%d' % (fold_k + 1, fold_ks))
        fold_k += 1

        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]

        fold_train_ious = []
        fold_train_loss = []
        fold_test_ious = []
        fold_test_loss = []

        fold_f1_score = 0

        for epoch in range(epochs):
            print('** Epoch %d/%d' % (epoch + 1, epochs))

            epoch_train_loss = 0
            epoch_train_iou = 0
            epoch_test_loss = 0
            epoch_test_iou = 0

            for X_batch, Y_batch in zip(X_train, Y_train):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                model.train()  # train mode

                # set parameter gradients to zero
                opt.zero_grad()

                # forward
                Y_pred = model(X_batch).squeeze()
                # print(Y_batch.shape, Y_pred.shape)
                loss = loss_fn(Y_pred, Y_batch)  # forward-pass
                # print(loss.shape)
                loss.backward()  # backward-pass
                opt.step()  # update weights

                model.eval()
                with torch.no_grad():
                    # calculate train iou:
                    Y_pred_thr = torch.sigmoid(Y_pred).greater_equal(threshold).to(torch.float)
                    iou = iou_pytorch(Y_pred_thr, Y_batch)

                epoch_train_iou += iou
                epoch_train_loss += loss

            epoch_train_iou /= len(X_train)
            fold_train_ious.append(epoch_train_iou.item())

            epoch_train_loss /= len(X_train)
            fold_train_loss.append(epoch_train_loss.item())

            print('train iou: %f' % epoch_train_iou)
            print('train loss: %f' % epoch_train_loss)

            # show intermediate results
            model.eval()  # testing mode
            with torch.no_grad():
                i = 0
                for X_batch, Y_batch in zip(X_test, Y_test):
                    X_batch = X_batch.to(device)
                    Y_batch = Y_batch.to(device)
                    Y_pred_val = model(X_batch).squeeze()
                    epoch_test_loss += loss_fn(Y_pred_val, Y_batch)  # forward-pass
                    Y_pred_val = torch.sigmoid(Y_pred_val).greater_equal(threshold).to(torch.float)
                    #  Если это последняя эпоха, то нарисовать картиночки и посчитать f1_score
                    if epoch == epochs - 1:
                        Y_batch = Y_batch.cpu()
                        Y_pred_val = Y_pred_val.cpu()
                        fold_f1_score += f1_score(Y_batch, Y_pred_val)

                        if i in draw_for:
                            initial_data = data_by_id_list[test_index[i]]
                            time_ticks = initial_data["time"].values
                            rr_initial = initial_data["x"].values
                            draw_pics(time_ticks, rr_initial, Y_pred_val, Y_batch)

                    i += 1

                    epoch_test_iou += iou_pytorch(Y_pred_val, Y_batch)

            epoch_test_iou /= len(X_test)
            fold_test_ious.append(epoch_test_iou.item())

            epoch_test_loss /= len(X_test)
            fold_test_loss.append(epoch_test_loss.item())

            print('test iou: %f' % epoch_test_iou)
            print('test loss: %f' % epoch_test_loss)

            lr_decay.step()

        all_train_ious.append(fold_train_ious)
        all_test_ious.append(fold_test_ious)
        all_train_loss.append(fold_train_loss)
        all_test_loss.append(fold_test_loss)

        fold_f1_score /= len(test_index)
        f1_scores.append(fold_f1_score)

    print("--------")

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16,16))

    for i in range(fold_ks):
        print(f"Fold {i+1}/{fold_ks}:")

        train_loss = all_train_loss[i]
        test_loss = all_test_loss[i]

        train_iou = all_train_ious[i]
        test_iou = all_test_ious[i]

        print(f"Last train IOU: {train_iou[-1]}")
        print(f"Last test IOU: {test_iou[-1]}")
        print(f"Average test F1 score: {f1_scores[i]}")

        axs[0].plot(train_loss, label=f"train loss {i}")
        axs[0].plot(test_loss, label=f"test loss {i}")
        axs[1].plot(train_iou, label=f"train IOU {i}")
        axs[1].plot(test_iou, label=f"test IOU {i}")

    axs[0].set_title("Loss")
    axs[1].set_title("IOU")

    axs[0].set_xlabel('epoch')
    axs[1].set_xlabel('epoch')

    axs[0].set_ylabel('Loss')
    axs[1].set_ylabel('IOU')

    plt.legend()
    plt.show()


def run_for_predict(model, opt, loss_fn, lr_decay, epochs, X, Y, device, to_train=True, save_to=None):

    threshold = 0.5

    test_data = pd.read_csv("data/CardioSpike/data/test.csv")
    test_observations = test_data["id"].unique()
    print(f"Test observations: {len(test_observations)}")

    test_data_by_id = test_data.groupby(['id', ])

    test_data_by_id_list = [test_data_by_id.get_group(x) for x in test_data_by_id.groups]

    X_test = []

    for sample_data in test_data_by_id_list:
        sample_data_increments = sample_data["x"][1:].values - sample_data["x"][:-1].values
        sample_data_derivatives = sample_data_increments / sample_data["x"][1:].values
        x = torch.from_numpy(sample_data_derivatives).view(1, 1, -1).double()
        X_test.append(x)

    X_test = np.array(X_test, dtype=object)

    if to_train:
        train_ious = []
        train_loss = []
        for epoch in range(epochs):
            print('** Epoch %d/%d' % (epoch + 1, epochs))

            epoch_train_iou = 0
            epoch_train_loss = 0

            for X_batch, Y_batch in zip(X, Y):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                model.train()  # train mode

                # set parameter gradients to zero
                opt.zero_grad()

                # forward
                Y_pred = model(X_batch).squeeze()
                # print(Y_batch.shape, Y_pred.shape)
                loss = loss_fn(Y_pred, Y_batch)  # forward-pass
                # print(loss.shape)
                loss.backward()  # backward-pass
                opt.step()  # update weights

                model.eval()
                with torch.no_grad():
                    # calculate train iou:
                    Y_pred_thr = torch.sigmoid(Y_pred).greater_equal(threshold).to(torch.float)
                    iou = iou_pytorch(Y_pred_thr, Y_batch)

                epoch_train_iou += iou
                epoch_train_loss += loss

            lr_decay.step()
            epoch_train_iou /= len(X)
            print(f"train IoU: {epoch_train_iou}")
            epoch_train_loss /= len(X)
            print(f"train loss: {epoch_train_loss}")
            train_ious.append(epoch_train_iou.item())
            train_loss.append(epoch_train_loss.item())

    else:
        model = CRNN().double()
        model.load_state_dict(torch.load(save_to))
        model = model.to(device)

    model.eval()  # testing mode
    test_data["y"] = 0
    index2data_id = list(test_data_by_id.groups)
    with torch.no_grad():
        i = 0
        for X_batch in X_test:
            X_batch = X_batch.to(device)
            Y_pred_val = model(X_batch).squeeze()
            Y_pred_val = torch.sigmoid(Y_pred_val).greater_equal(threshold).to(torch.float)
            Y_pred_val = Y_pred_val.cpu().numpy()

            Y_pred_val = np.concatenate((np.zeros(1), Y_pred_val)).astype(int)

            id = index2data_id[i]
            test_data.loc[test_data["id"] == id, "y"] = Y_pred_val

            initial_data = test_data_by_id_list[i]
            time_ticks = initial_data["time"].values
            rr_initial = initial_data["x"].values
            draw_test(time_ticks, rr_initial, Y_pred_val)
            i += 1

    test_data = test_data.drop("x", 1)
    test_data.to_csv("submission2.csv", index=False)

    if not to_train:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(16,16))

        axs[0].plot(train_loss, label=f"train loss")
        axs[1].plot(train_ious, label=f"train IOU")

        axs[0].set_title("Loss")
        axs[1].set_title("IOU")

        axs[0].set_xlabel('epoch')
        axs[1].set_xlabel('epoch')

        axs[0].set_ylabel('Loss')
        axs[1].set_ylabel('IOU')

        plt.legend()
        plt.show()

        if save_to is not None:
            torch.save(model.state_dict(), save_to)

        # Load:
        #
        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        # model.eval()


def main(predict=False, save_to=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = modelClass().double()
    max_epochs = 150
    opt = optim.Adam(model.parameters(), lr=1e-2)
    loss_crit = nn.BCEWithLogitsLoss()
    lr_decay = optim.lr_scheduler.ExponentialLR(
        optimizer=opt, gamma=0.99)

    model = model.to(device)

    if predict:
        run_for_predict(model, opt, loss_crit, lr_decay, max_epochs, X, Y, device, to_train=False, save_to=save_to)
    else:
        cross_validation(
            model, opt, loss_crit, lr_decay, max_epochs, X, Y, device, pics_num)


def test():
    model = modelClass().double()

    print(X[0].shape, X[0].type())
    Y_pred = model(X[0])
    print(Y_pred.shape)


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("--test", action="store_true", default=False)
    parser.add_option("--predict", action="store_true", default=False)
    parser.add_option("--crnn", action="store_true", default=False)
    parser.add_option("--picsnum", action="store", type=int)
    parser.add_option("--save_to", action="store", type=str)

    (opts, args) = parser.parse_args()

    # read data
    data = pd.read_csv("data/CardioSpike/data/train.csv")
    observations = data["id"].unique()
    print(f"Observations: {len(observations)}")

    data_by_id = data.groupby(['id', ])

    data_by_id_list = [data_by_id.get_group(x) for x in data_by_id.groups]

    X = []
    Y = []

    for sample_data in data_by_id_list:
        sample_data_increments = sample_data["x"][1:].values - sample_data["x"][:-1].values
        sample_data_derivatives = sample_data_increments / sample_data["x"][1:].values
        x = torch.from_numpy(sample_data_derivatives).view(1, 1, -1).double()
        X.append(x)
        y = torch.from_numpy(sample_data["y"][1:].values).double()  # .view(1, -1)
        Y.append(y)

    X = np.array(X, dtype=object)
    Y = np.array(Y, dtype=object)

    pics_num = opts.picsnum

    if opts.crnn:
        modelClass = CRNN
    else:
        modelClass = SimpleNet

    if opts.test:
        test()
    else:
        main(predict=opts.predict, save_to=opts.save_to)
