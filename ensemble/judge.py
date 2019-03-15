import os
import numpy as np
from models import model
from sklearn.model_selection import train_test_split
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from dataset import load_dataset
import os
from os import listdir
from os.path import isfile, join, split
from ensemble.visualisation import ranging
import pickle as pkl
from metrics import statistics, F_score
from train import train
from metrics import statistics, F_score
from dataset import load_dataset
from utils import path_to_ensemble_models

def train_judge(model_paths_list, xtest, xtrain, ytest, ytrain, name):
    """
    trains the judicial network
    :param model_paths_list: list of paths to the saved ensemble members
    :param xtest: validation data
    :param ytest: validation data
    :return:
    """
    win_len = 3072
    batch_size=10
    epochs = 14
    tmp_model = load_model(model_paths_list[0])
    ens_predict_test = np.array(tmp_model.predict(xtest[:,:4992,:])[:,:,:3])
    ens_predict_train = np.array(tmp_model.predict(xtrain[:,:4992,:])[:,:,:3])

    for path in model_paths_list[1:]:
        tmp_model = load_model(path)
        predict = np.array(tmp_model.predict(xtest[:,:4992,:])[:,:,:3])
        ens_predict_test = np.concatenate((ens_predict_test,predict), axis=2)

        predict = np.array(tmp_model.predict(xtrain[:,:4992,:])[:,:,:3])
        ens_predict_train = np.concatenate((ens_predict_train,predict), axis=2)

    xtest = np.concatenate((xtest[:,:4992,0:1], ens_predict_test), axis=2)
    xtrain = np.concatenate((xtrain[:,:4992,0:1], ens_predict_train), axis=2)


    judge_model = model.make_model(num_leads_signal=len(model_paths_list)*3+1)

    train(judge_model,
          model_name= "judge_model",
          x_test=xtest,
          x_train=xtrain,
          y_test=ytest,
          y_train=ytrain,
          win_len=win_len,
          batch_size=batch_size,
          epochs=epochs,  folder_name= path_to_ensemble_models)

    return judge_model

if __name__ == "__main__":
    from ensemble.make_ensemble import ensemble_predict, ensemble_predict_with_judge
    from utils import path_to_ensemble_data, path_to_ensemble_models

    win_len = 3072
    path_to_models = path_to_ensemble_models
    path_to_data = path_to_ensemble_data

    model_paths_list = [join(path_to_models,f) for f in listdir(path_to_models) if isfile(join(path_to_models, f))]
    xy = load_dataset()
    X = xy["x"]
    Y = xy["y"]
    xtrain, xtest, ytrain, ytest= train_test_split(X, Y, test_size=0.33, random_state=42)

    train_judge(model_paths_list, xtest, xtrain, ytest, ytrain, 0)

    pred_j = ensemble_predict_with_judge(model_paths_list, path_to_ensemble_models+'\\judge_model0', xtest)
    pred_e = ensemble_predict(model_paths_list, xtest)

    stat_j = statistics(ytest[:,win_len//2:5000-win_len//2,:], pred_j[:,win_len//2:5000-win_len//2,:])

    print("ensemble with judge:")
    print(stat_j)

    stat_e = statistics(ytest[:,win_len//2:5000-win_len//2,:], pred_e[:,win_len//2:5000-win_len//2,:])
    print("simple ensemble:")
    print(stat_e)
    #ranging(pred_e, xtest, ytest, win_len, col= "k", is_path = False)
    #plt.savefig("ensmodel"+str(i)+".jpg")
    #plt.clf()