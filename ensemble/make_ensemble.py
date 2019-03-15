import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from dataset import load_dataset
from os import listdir
from os.path import isfile, join, split
from ensemble.visualisation import ranging, plot_two_prediction
from metrics import statistics, F_score

def ensemble_predict(model_paths_list, x):
    """
    makes the annotation an ECG by ensemble
    :param model_paths_list: list of paths to the saved ensemble members
    :param x: dataset
    :return: predicted annotation
    """
    model = load_model(model_paths_list[0])
    ens_predict = np.array(model.predict(x))

    for path in model_paths_list[1:]:
        model = load_model(path)
        predict = np.array(model.predict(x))
        ens_predict = predict + ens_predict

    ens_predict = ens_predict/len(model_paths_list)
    return ens_predict

def ensemble_predict_with_judge(model_paths_list, judge_path, x):
    """
    makes the annotation an ECG by ensemble with judge
    :param model_paths_list: list of paths to the saved ensemble members
    :param x: dataset
    :param judge_path: path to judge model
    :return: predicted annotation
    """
    model = load_model(model_paths_list[0])
    ens_predict = np.array(model.predict(x[:,:4992,:])[:,:,:3])

    for path in model_paths_list[1:]:
        model = load_model(path)
        predict = np.array(model.predict(x[:,:4992,:])[:,:,:3])
        ens_predict = np.concatenate((ens_predict,predict), axis=2)

    xtest = np.concatenate((x[:,:4992,0:1], ens_predict), axis=2)

    judge_model = load_model(judge_path)
    predict = judge_model.predict(xtrain)
    return predict

def histogram(model_paths_list, x, y, win_len, threshold = 0.99):
    """
    returns a dictionary: {model number: number of patients from x with F1 score > threshold}
    :param model_paths_list: list of paths to the saved models
    :param x: dataset
    :param y: GT annotation
    """
    dict = {}
    for path in model_paths_list:
        _, filename = split(path)
        model_num = int(filename[len("ens_model_"):-3])
        dict[model_num] = 0
        model = load_model(path)
        predict = np.array(model.predict(x))
        for i in range(len(x)):
            pred = predict[i,win_len//2:5000-win_len//2,:]
            y_i = y[i,win_len//2:5000-win_len//2,:]
            stat = statistics(np.expand_dims(y_i, axis=0), np.expand_dims(pred, axis=0))
            F = F_score(stat)
            if F >=threshold:
                dict[model_num] += 1

    return dict

if __name__ == "__main__":
    from utils import path_to_ensemble_data, path_to_ensemble_models
    win_len = 3072

    model_paths_list = [join(path_to_ensemble_models,f) for f in listdir(path_to_ensemble_models) if isfile(join(path_to_ensemble_models, f))]

    xy = load_dataset()

    X = xy["x"]
    Y = xy["y"]
    offsets = (5000 - win_len)//2
    xtrain, xtest, ytrain, ytest= train_test_split(X, Y, test_size=0.33, random_state=42)
    model = load_model(path_to_ensemble_models+"\\ens_model_1.h5")
    pred_e = ensemble_predict(model_paths_list, xtest)
    pred_ = model.predict(xtest)

    stat = statistics(ytest[:,win_len//2:5000-win_len//2,:], pred_e[:,win_len//2:5000-win_len//2,:])
    print(F_score(stat))

    #stat.to_csv("stats_one_test.csv", sep = ';')
    ranging(pred_e, xtest, ytest, win_len, col= "k", is_path = False)
    plt.show()

    dict = histogram(model_paths_list, xtrain, ytrain, win_len, threshold = 0.99)
    plt.bar(list(dict.keys()), dict.values(), color='g', alpha = 0.5)
    plt.show()


    plot_two_prediction(pred_e, pred_, xtest, ytest, win_len, [5])