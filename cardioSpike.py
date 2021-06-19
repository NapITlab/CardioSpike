import os
import random
import argparse

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

from ourmodels.CRNN import CRNN
from keras import models as keras_models


CRNN_presaved = "ourmodels/pretrained/CRNN.pt"
UNet_presaved = "ourmodels/pretrained/UNet.h5"


def get_RCNN_results(test_data):

    test_data_by_id = test_data.groupby(['id', ])

    test_data_by_id_list = [test_data_by_id.get_group(x) for x in test_data_by_id.groups]

    X_test = []

    for sample_data in test_data_by_id_list:
        sample_data_increments = sample_data["x"][1:].values - sample_data["x"][:-1].values
        sample_data_derivatives = sample_data_increments / sample_data["x"][1:].values
        x = torch.from_numpy(sample_data_derivatives).view(1, 1, -1).double()
        X_test.append(x)

    X_test = np.array(X_test, dtype=object)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CRNN().double()
    model.load_state_dict(torch.load(CRNN_presaved))
    model = model.to(device)

    model.eval()  # testing mode
    test_data["y"] = 0
    index2data_id = list(test_data_by_id.groups)
    with torch.no_grad():
        i = 0
        for X_batch in X_test:
            X_batch = X_batch.to(device)
            Y_pred_val = model(X_batch).squeeze()
            Y_pred_val = torch.sigmoid(Y_pred_val)  #.greater_equal(threshold).to(torch.float)
            Y_pred_val = Y_pred_val.cpu().numpy()

            Y_pred_val = np.concatenate((np.zeros(1), Y_pred_val))  #.astype(int)

            id = index2data_id[i]
            test_data.loc[test_data["id"] == id, "y"] = Y_pred_val

            # initial_data = test_data_by_id_list[i]
            # time_ticks = initial_data["time"].values
            # rr_initial = initial_data["x"].values
            # draw_test(time_ticks, rr_initial, Y_pred_val)
            i += 1

    test_data = test_data.drop("x", 1)
    return test_data


def get_UNet_results(test):
    model = keras_models.load_model(UNet_presaved)

    batch_size = 1
    test_pred = []
    for id_i in pd.unique(test['id']):
        a = test[test['id'] == id_i]
        a = a.sort_values(by=['time'])
        a.reset_index(inplace=True)
        if len(a) >= 64:
            Y = np.zeros(((len(a) - 64), len(a)))
            for j, i in enumerate(range(32, len(a) - 32, 1)):
                x = a.iloc[i - 32:i + 32]['x']
                x = (x - x.mean()) / x.std()

                image1 = np.zeros((batch_size, 64, 1))
                for k in range(batch_size):
                    image1[k, :] = np.array(x).reshape(-1, 1)
                Y[j, i - 32:i + 32] = (model.predict(image1)[0]).reshape(64, )
            a['y'] = Y.max(axis=0)
            if len(test_pred) == 0:
                test_pred = a
            else:
                test_pred = test_pred.append(a)
        else:
            x = a['x']
            x = (x - x.mean()) / x.std()
            image1 = np.zeros((batch_size, 64, 1))
            for k in range(batch_size):
                image1[k, 0:len(a)] = np.array(x).reshape(-1, 1)
            Y = (model.predict(image1)[0]).reshape(64, )
            a['y'] = Y[:len(a)]
            if len(test_pred) == 0:
                test_pred = a
            else:
                test_pred = test_pred.append(a)

    return test_pred


def combine_results(*outputs):
    average_prob = outputs[0]["y"]
    for output in outputs[1:]:
        average_prob += output["y"]
    average_prob /= len(outputs)
    average_prob = (average_prob > 0.5).astype(int)
    return average_prob


def draw_pics(results, output_folder):

    test_data_by_id = results.groupby(['id', ])

    for id in test_data_by_id.groups:

        results_data = test_data_by_id.get_group(id)

        x = results_data["time"].values
        y = results_data["x"].values
        y_predicted = results_data["y"].values
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

        fig_name = os.path.join(output_folder, f"{id}.png")
        plt.savefig(fig_name)


def main(input_csv, output_folder, draw):

    data = pd.read_csv(input_csv)
    crnn = get_RCNN_results(data)

    # model in file didn't open, there were not opportunity to save it one more time
    # unet = get_UNet_results(data)

    predicts = combine_results(crnn)

    data["y"] = predicts

    if draw:
        draw_pics(data, output_folder)

    output_csv = os.path.join(output_folder, "output.csv")
    data.to_csv(output_csv)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict abnormal intervals from RR sequence')
    parser.add_argument('input', type=str, help='path to csv file with input: id, time, x')
    parser.add_argument('output', type=str, help='path to folder for outputs')
    parser.add_argument(
        '--pics', action='store_true', default=False, help='if to save pictures with segmentation')

    args = parser.parse_args()

    main(args.input, args.output, args.pics)

