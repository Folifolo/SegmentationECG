import os
from os import listdir
from os.path import isfile, join
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split

from dataset import load_dataset
from metrics import statistics, F_score


def draw_one(model_path, x, y, patients, win_len):
    """
    print F1_score, plot ECG annotation of the network and ground true
    :param model_path: path to the trained model
    :param x: array of ECG
    :param y: array of annotation
    :param pacients: list of patients numbers to be plotted
    """
    for pacient in patients:
        offsets = (5000 - win_len)//2
        model = load_model(model_path)
        X = np.expand_dims(x[pacient, :, :], axis=0)
        Y = np.expand_dims(y[pacient,offsets:5000 - offsets,:], axis=0)

        prediction = np.array(model.predict(X))
        prediction = prediction[:,offsets:5000-offsets,:]

        x_axis = np.arange(offsets/500, (win_len +offsets)/500, 1/500)
        plt.figure(figsize=(20, 5))
        plt.plot(x_axis, x[pacient, offsets:5000 - offsets, 0], 'k')

        predict_rounded = np.argmax(prediction, axis=2)[pacient]
        one_hot = np.zeros((predict_rounded.size, predict_rounded.max()+1))
        one_hot[np.arange(predict_rounded.size), predict_rounded] = 1

        plt.fill_between(x_axis, Y[0, :win_len, 1]*40 + -50, -50, color='r', alpha=0.3)
        plt.fill_between(x_axis, Y[0, :win_len, 2]*40 + -50, -50, color='g', alpha=0.3)
        plt.fill_between(x_axis, Y[0, :win_len, 0]*40 + -50, -50, color='b', alpha=0.3)
        plt.fill_between(x_axis, list(one_hot[:win_len, 1]*40), 0, color='r', alpha=0.3)
        plt.fill_between(x_axis, list(one_hot[:win_len, 2]*40), 0, color='g', alpha=0.3)
        plt.fill_between(x_axis, list(one_hot[:win_len, 0]*40), 0, color='b', alpha=0.3)


        stat = statistics(Y, prediction)
        F = F_score(stat)
        print(stat)
        print(F)
        plt.show()
        #plt.savefig("ill"+str(i)+".png")
        #plt.clf()

def draw_all(list_of_models, x, y, patients, win_len):
    """
    plot the output of all networks from the list on a single image
    :param list_of_models: list of paths to the trained models
    :param x: array of ECG
    :param y: array of annotation
    :param pacients: list of patients numbers to be plotted
    """
    offsets = (5000 - win_len)//2
    X = x
    Y = y[:,offsets:5000 - offsets,:]

    x_axis = np.arange(offsets/500, (win_len +offsets)/500, 1/500)
    for i in patients:
        plt.figure(figsize=(20, 5))
        off = 30
        for model_path in list_of_models:
            off +=50
            model = load_model(model_path)
            prediction = np.array(model.predict(X))
            prediction = prediction[:,offsets:5000-offsets,:]
            plt.plot(x_axis, x[i, offsets:5000 - offsets, 0], 'k')
            predict_rounded = np.argmax(prediction, axis=2)[i]
            one_hot = np.zeros((predict_rounded.size, predict_rounded.max()+1))
            one_hot[np.arange(predict_rounded.size), predict_rounded] = 1

            plt.fill_between(x_axis, Y[i, :win_len, 1]*60 + -50, -50, color='r', alpha=0.3)
            plt.fill_between(x_axis, Y[i, :win_len, 2]*60 + -50, -50, color='g', alpha=0.3)
            plt.fill_between(x_axis, Y[i, :win_len, 0]*60 + -50, -50, color='b', alpha=0.3)

            plt.xlim((3,8))
            plt.plot(x_axis, list(prediction[i,:, 1]*40+off), off, color='r', alpha=0.3)
            plt.plot(x_axis, list(prediction[i,:, 2]*40+off), off, color='g', alpha=0.3)
            plt.plot(x_axis, list(prediction[i,:, 0]*40+off), off, color='b', alpha=0.3)

        plt.show()
        #plt.savefig("ill"+str(i)+".png")
        #plt.clf()


def plot_two_prediction(pred1, pred2, x, y, win_len, patients):
    """
    plot the answers of the two models
    :param pred1: prediction of first model
    :param pred2: prediction of second model
    """
    for pacient in patients:
        offsets = (5000 - win_len)//2
        X = x[pacient, :, :]
        Y = y[pacient,offsets:5000 - offsets,:]

        x_axis = np.arange(offsets/500, (win_len +offsets)/500, 1/500)

        plt.figure(figsize=(20, 5))
        plt.plot(x_axis, X[offsets:5000 - offsets, 0], 'k')

        pred1 = pred1[:,offsets:5000-offsets,:]
        predict1_rounded = np.argmax(pred1, axis=2)[pacient]
        one_hot1 = np.zeros((predict1_rounded.size, predict1_rounded.max()+1))
        one_hot1[np.arange(predict1_rounded.size), predict1_rounded] = 1

        pred2 = pred2[:,offsets:5000-offsets,:]
        predict2_rounded = np.argmax(pred2, axis=2)[pacient]
        one_hot2 = np.zeros((predict2_rounded.size, predict2_rounded.max()+1))
        one_hot2[np.arange(predict2_rounded.size), predict2_rounded] = 1

        plt.fill_between(x_axis, Y[:win_len, 1]*40 + -50, -50, color='r', alpha=0.3)
        plt.fill_between(x_axis, Y[:win_len, 2]*40 + -50, -50, color='g', alpha=0.3)
        plt.fill_between(x_axis, Y[:win_len, 0]*40 + -50, -50, color='b', alpha=0.3)

        plt.fill_between(x_axis, list(one_hot1[:win_len, 1]*40), 0, color='r', alpha=0.3)
        plt.fill_between(x_axis, list(one_hot1[:win_len, 2]*40), 0, color='g', alpha=0.3)
        plt.fill_between(x_axis, list(one_hot1[:win_len, 0]*40), 0, color='b', alpha=0.3)

        plt.fill_between(x_axis, list(one_hot2[:win_len, 1]*40+50), 50, color='r', alpha=0.3)
        plt.fill_between(x_axis, list(one_hot2[:win_len, 2]*40+50), 50, color='g', alpha=0.3)
        plt.fill_between(x_axis, list(one_hot2[:win_len, 0]*40+50), 50, color='b', alpha=0.3)

        plt.show()
        plt.clf()

def ranging(model_path, x, y, win_len, col= "k", model_pred = None):
    """
    plot a scattergram of F1 score for each patient
    :return: list of F1 scores
    """
    offsets = (5000 - win_len)//2
    Y = y[:,offsets:5000 - offsets,:]

    if model_pred == None:
        model = load_model(model_path)
        prediction = np.array(model.predict(x))
    else:
        prediction = model_pred
    prediction = prediction[:,offsets:5000-offsets,:]

    dict = {}
    for i in range(len(x)):
        prediction_i = prediction[i,:,:]
        y_i = Y[i,:,:]
        stat = statistics(np.expand_dims(y_i, axis=0), np.expand_dims(prediction_i, axis=0))
        F = F_score(stat)
        dict[i] = F

    dict = sorted(dict.items())
    x, y_i = zip(*dict)
    plt.scatter(x, y_i, c=col, alpha=0.3)
    plt.show()
    return y_i

def split_data(xy1, xy2):
    """
    removes the elements xy1, xy2 found in
    """
    x1 = xy1["x"]
    y1 = xy1["y"]
    x2 = xy2["x"]
    for i in range(len(x2)):
        for j in range(len(x1)):
            if np.array_equal(x1[j,:,:],x2[i,:,:]):
                x1 = np.delete(x1, j, 0)
                y1 = np.delete(y1, j, 0)
                break
    return x1, y1

def print_class(path1, path2):
    """
    plot ECGs that are contained in path1 but not in path2
    """
    offsets = (5000 - win_len)//2
    pkl_filename = path1
    if os.path.exists(pkl_filename):
        infile = open(pkl_filename, 'rb')
        xy1 = pkl.load(infile)
        infile.close()

    pkl_filename = path2
    if os.path.exists(pkl_filename):
        infile = open(pkl_filename, 'rb')
        xy2 = pkl.load(infile)
        infile.close()
    x, y = split_data(xy1, xy2)

    x_axis = np.arange(offsets/500, (win_len +offsets)/500, 1/500)
    y= y[:,offsets:5000 - offsets,:]

    for i in range(len(x)):
        plt.figure(figsize=(20, 5))
        plt.plot(x_axis, x[i, offsets:5000 - offsets, 0], 'k')

        plt.fill_between(x_axis, y[i, :, 1]*60 + -50, -50, color='r', alpha=0.3)
        plt.fill_between(x_axis, y[i, :, 2]*60 + -50, -50, color='g', alpha=0.3)
        plt.fill_between(x_axis, y[i, :, 0]*60 + -50, -50, color='b', alpha=0.3)

        plt.show()
        #plt.savefig("class0"+"_"+str(i)+".png")
        #plt.clf()



if __name__ == "__main__":
    from utils import path_to_ensemble_data, path_to_ensemble_models
    #usage example
    win_len = 3072
    leads = 12

    model_paths_list = [join(path_to_ensemble_models,f) for f in listdir(path_to_ensemble_models) if isfile(join(path_to_ensemble_models, f))]

    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]

    draw_one(path_to_ensemble_models+"\\ens_model_1.h5", X, Y, [10], win_len)
    draw_all(model_paths_list, X, Y, [10], win_len)
    ranging(model_paths_list[0], X, Y, win_len, col= "k")
    print_class(path_to_ensemble_data+"\\trim_0.pkl",path_to_ensemble_data+"\\trim_1.pkl")
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.33, random_state=42)

    #print_class(xtest, path_to_data+"\\trim_12.pkl")

