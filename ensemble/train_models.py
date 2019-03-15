import os
import numpy as np
from models import model
from sklearn.model_selection import train_test_split
import pickle as pkl

from train import train
from metrics import statistics, F_score
from dataset import load_dataset



def trim(model, xtrain, ytrain, name, threshold, path_to_data, win_len):
    """
    removes from xtrain, ytrain elements on which the model has F1 greater than threshold
    :param path_to_data: path to the folder where the trimmed dataset will be saved
    :return: trimmed dataset
    """
    pred_train = np.array(model.predict(xtrain))
    xtrain_new = xtrain.copy()
    ytrain_new = ytrain.copy()
    counter = 0
    for i in range(len(xtrain)):
        pred = pred_train[i,win_len//2:5000-win_len//2,:]
        y = ytrain[i,win_len//2:5000-win_len//2,:]
        stat = statistics(np.expand_dims(y, axis=0), np.expand_dims(pred, axis=0))
        F = F_score(stat)
        if F >=threshold:
            xtrain_new = np.delete(xtrain_new, i-counter, axis = 0)
            ytrain_new = np.delete(ytrain_new, i-counter, axis = 0)
            counter+=1

    if not os.path.exists(path_to_data):
        os.makedirs(path_to_data)
    outfile = open(path_to_data + "\\trim_" + name + ".pkl", 'wb')
    pkl.dump({"x":xtrain_new, "y":ytrain_new}, outfile)
    outfile.close()
    return xtrain_new, ytrain_new

if __name__ == "__main__":
    from utils import path_to_ensemble_data, path_to_ensemble_models

    win_len = 3072
    batch_size=10
    epochs = 14
    name = 0
    threshold = 0.99
    load = False  #load by name if training has been interrupted

    xy = load_dataset()

    if load == True:
        pkl_filename = path_to_ensemble_data + "\\trim_" + str(name) + ".pkl"
        if os.path.exists(pkl_filename):
            infile = open(pkl_filename, 'rb')
            xy = pkl.load(infile)
            infile.close()

    X = xy["x"]
    Y = xy["y"]

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.33, random_state=42)

    length = len(xtrain)
    while len(xtrain)> 2:
        name += 1
        tmp_model = model.make_model()
        train(tmp_model,
              model_name= "ens_model_" + str(name),
              x_test=xtest,
              x_train=xtrain,
              y_test=ytest,
              y_train=ytrain,
              win_len=win_len,
              batch_size=batch_size,
              epochs=epochs,  folder_name= path_to_ensemble_models)

        xtrain, ytrain = trim(tmp_model, xtrain, ytrain, str(name), threshold, path_to_ensemble_data, win_len)
        print("in model " + str(name) + "  patients left: " + str(len(xtrain)))
        if len(xtrain) == length:
            name -= 1
        length = len(xtrain)