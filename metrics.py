import numpy as np
import tensorflow as tf
import pandas as pd
from dataset import FREQUENCY_OF_DATASET

freq = FREQUENCY_OF_DATASET
tolerance = (150 / 1000) * freq * 60


def change_mask(sample):
    """
    bring the mask to the form where only the start and end of the interval is stored
    :param sample: annotation of ecg
    :return: lists of starts and ends
    """
    p = [[], []]
    qrs = [[], []]
    t = [[], []]
    for i in range(sample.shape[0] - 1):
        if sample[i] != 0 and sample[i + 1] == 0:
            # start_P
            p[0].append(i)
        elif sample[i] != 1 and sample[i + 1] == 1:
            # start_QRS
            qrs[0].append(i)
        elif sample[i] != 2 and sample[i + 1] == 2:
            # start_T
            t[0].append(i)

        if sample[i] == 0 and sample[i + 1] != 0:
            # end_P
            p[1].append(i)
        elif sample[i] == 1 and sample[i + 1] != 1:
            # end_QRS
            qrs[1].append(i)
        elif sample[i] == 2 and sample[i + 1] != 2:
            # end_T
            t[1].append(i)
    return p, qrs, t


def comprassion(mask1, mask2, start_or_end):
    """
    compares two annotations
    :param mask1: annotation by algorithm
    :param mask2: ground truth annotation
    :param start_or_end: 0 -- start of interval, 1 -- end of interval
    :return: lists of positions true positive answers, false positive, false negative and variances in tp
    """
    tp = []
    fp = []
    fn = []
    error = []

    pulse = 0

    for count in range(len(mask2[start_or_end])-1):
        pulse += mask2[start_or_end][count+1] - mask2[start_or_end][count]
    if pulse == 0:
        pulse = 70
    else:
        pulse =1/(pulse / (count+1) /500)*60
    for p_1 in mask1[start_or_end]:
        flag = False
        for p_2 in mask2[start_or_end]:
            if p_1 + (tolerance/pulse) >= p_2 > p_1 - (tolerance/pulse):
                tp.append(p_1)
                error.append((p_1 - p_2) / freq)
                mask2[start_or_end].remove(p_2)
                flag = True
                break
        if not flag:
            fp.append(p_1)

    for p_2 in mask2[start_or_end]:
        fn.append(p_2)

    return tp, fp, fn, error


class Metrics(object):
    """
    class of Se and PPV metrics for tensorflow
    """
    def Se(self, y_true, y_pred):
        return tf.py_func(self._np_Se, [y_true, y_pred], tf.float32)

    @staticmethod
    def _np_Se(y_true, y_pred):
        true_pos = 0
        false_neg = 0

        for j in range(y_pred.shape[0]):
            sample_true = np.argmax(y_true[j], 1)
            sample_pred = np.argmax(y_pred[j], 1)

            p1, qrs1, t1 = change_mask(sample_pred)
            p2, qrs2, t2 = change_mask(sample_true)

            for i in range(2):
                tp, _, fn, _ = comprassion(p1, p2, i)
                true_pos += len(tp)
                false_neg += len(fn)

                tp, _, fn, _ = comprassion(qrs1, qrs2, i)
                true_pos += len(tp)
                false_neg += len(fn)

                tp, _, fn, _ = comprassion(t1, t2, i)
                true_pos += len(tp)
                false_neg += len(fn)

                if true_pos + false_neg == 0:
                    res = 0
                else:
                    res = true_pos / (true_pos + false_neg)

        return np.mean(res).astype(np.float32)

    def PPV(self, y_true, y_pred):
        return tf.py_func(self._np_PPV, [y_true, y_pred], tf.float32)

    @staticmethod
    def _np_PPV(y_true, y_pred):
        true_pos = 0
        false_pos = 0

        for j in range(y_pred.shape[0]):
            sample_true = np.argmax(y_true[j], 1)
            sample_pred = np.argmax(y_pred[j], 1)

            p1, qrs1, t1 = change_mask(sample_pred)
            p2, qrs2, t2 = change_mask(sample_true)

            for i in range(2):
                tp, fp, _, _ = comprassion(p1, p2, i)
                true_pos += len(tp)
                false_pos += len(fp)

                tp, fp, _, _ = comprassion(qrs1, qrs2, i)
                true_pos += len(tp)
                false_pos += len(fp)

                tp, fp, _, _ = comprassion(t1, t2, i)
                true_pos += len(tp)
                false_pos += len(fp)

                if true_pos + false_pos == 0:
                    res = 0
                else:
                    res = true_pos / (true_pos + false_pos)

        return np.mean(res).astype(np.float32)


def statistics(y_true, y_pred):
    """
    calculate Sensitivity, PPV, mean error and variance of annotation
    :param y_true: ground truth annotation
    :param y_pred: annotation by algorithm
    :return: pandas df with statistic
    """
    df_res = pd.DataFrame(
        {'start_p': [0.0, 0.0, 0.0, 0.0], 'end_p': [0.0, 0.0, 0.0, 0.0], 'start_qrs': [0.0, 0.0, 0.0, 0.0],
         'end_qrs': [0.0, 0.0, 0.0, 0.0], 'start_t': [0.0, 0.0, 0.0, 0.0], 'end_t': [0.0, 0.0, 0.0, 0.0]},
        index=['Se', 'PPV', 'm', 'sigma^2'])
    df_stat = pd.DataFrame(
        {'start_p': [0, 0, 0], 'end_p': [0, 0, 0], 'start_qrs': [0, 0, 0], 'end_qrs': [0, 0, 0], 'start_t': [0, 0, 0],
         'end_t': [0, 0, 0]}, index=['tp', 'fp', 'fn'])
    df_errors = pd.DataFrame(
        {'start_p': [[]], 'end_p': [[]], 'start_qrs': [[]], 'end_qrs': [[]], 'start_t': [[]], 'end_t': [[]]})
    for j in range(y_pred.shape[0]):
        sample_true = np.argmax(y_true[j], -1)
        sample_pred = preproc(np.argmax(y_pred[j], -1))

        p1, qrs1, t1 = change_mask(sample_pred)
        p2, qrs2, t2 = change_mask(sample_true)

        tp, fp, fn, error = comprassion(p1, p2, 0)
        df_stat.at['tp', 'start_p'] += len(tp)
        df_stat.at['fp', 'start_p'] += len(fp)
        df_stat.at['fn', 'start_p'] += len(fn)
        df_errors.at[0, 'start_p'].extend(error)

        tp, fp, fn, error = comprassion(p1, p2, 1)
        df_stat.at['tp', 'end_p'] += len(tp)
        df_stat.at['fp', 'end_p'] += len(fp)
        df_stat.at['fn', 'end_p'] += len(fn)
        df_errors.at[0, 'end_p'].extend(error)

        tp, fp, fn, error = comprassion(qrs1, qrs2, 0)
        df_stat.at['tp', 'start_qrs'] += len(tp)
        df_stat.at['fp', 'start_qrs'] += len(fp)
        df_stat.at['fn', 'start_qrs'] += len(fn)
        df_errors.at[0, 'start_qrs'].extend(error)

        tp, fp, fn, error = comprassion(qrs1, qrs2, 1)
        df_stat.at['tp', 'end_qrs'] += len(tp)
        df_stat.at['fp', 'end_qrs'] += len(fp)
        df_stat.at['fn', 'end_qrs'] += len(fn)
        df_errors.at[0, 'end_qrs'].extend(error)

        tp, fp, fn, error = comprassion(t1, t2, 0)
        df_stat.at['tp', 'start_t'] += len(tp)
        df_stat.at['fp', 'start_t'] += len(fp)
        df_stat.at['fn', 'start_t'] += len(fn)
        df_errors.at[0, 'start_t'].extend(error)

        tp, fp, fn, error = comprassion(t1, t2, 1)
        df_stat.at['tp', 'end_t'] += len(tp)
        df_stat.at['fp', 'end_t'] += len(fp)
        df_stat.at['fn', 'end_t'] += len(fn)
        df_errors.at[0, 'end_t'].extend(error)


    for index in df_res.columns:
        for i in range(len(df_errors.at[0, index])):
            df_errors.at[0, index][i] = df_errors.loc[0, index][i]*1000
        if df_stat.loc['tp', index] == 0:
            if df_stat.loc['fn', index] == 0 and df_stat.loc['fp', index] != 0:
                df_res.at['Se', index] = 1
                df_res.at['PPV', index] = 0
                df_res.at['sigma^2', index] = 0
                df_res.at['m', index] = 0
            elif df_stat.loc['fn', index] != 0 and df_stat.loc['fp', index] == 0:
                df_res.at['Se', index] = 0
                df_res.at['PPV', index] = 1
                df_res.at['sigma^2', index] = 0
                df_res.at['m', index] = 0
            elif df_stat.loc['fn', index] != 0 and df_stat.loc['fp', index] != 0:
                df_res.at['Se', index] = 0
                df_res.at['PPV', index] = 0
                df_res.at['sigma^2', index] = 0
                df_res.at['m', index] = 0
            elif df_stat.loc['fn', index] == 0 and df_stat.loc['fp', index] == 0:
                df_res.at['Se', index] = 1
                df_res.at['PPV', index] = 1
                df_res.at['sigma^2', index] = 0
                df_res.at['m', index] = 0
        else:
            df_res.at['Se', index] = df_stat.loc['tp', index] / (df_stat.loc['tp', index] + df_stat.loc['fn', index])
            df_res.at['PPV', index] = df_stat.loc['tp', index] / (df_stat.loc['tp', index] + df_stat.loc['fp', index])
            df_res.at['sigma^2', index] = np.var(df_errors.loc[0, index])
            df_res.at['m', index] = np.mean(df_errors.loc[0, index])
    return df_res


def F_score(stat):
    """
    calculate F1 score by df table from statistics function
    :param stat: dataframe obtained from statistics function
    :return: F1 score
    """
    presision = 0
    recall = 0
    for index in stat.columns:
        recall += stat.loc['Se', index]
        presision += stat.loc['PPV', index]
    presision = presision/6
    recall = recall/6
    F = 2*((presision*recall)/(presision+recall))
    return F

def preproc(sample):
    """
    postprocessing of the prediction of the network: removes small breaks in intervals
    :param sample: ecg annotation
    :return: postprocessed annotation
    """
    for i in range(sample.shape[0]-10):
        if sample[i] !=0:
            if sample[i] != sample[i+1]:
                for j in range(10):
                    if sample[i+j+1] == sample[i]:
                        sample[i+1:i+j] = sample[i]
    return sample

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from dataset import load_dataset
    from keras.models import load_model
    from utils import path_to_ensemble_models

    xy = load_dataset(fixed_baseline=False)
    X = xy["x"]
    Y = xy["y"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    metric = Metrics()
    Se = metric.Se
    model = load_model(path_to_ensemble_models+'\\mymodel1.h5', custom_objects={'Se': Se})

    pred_test = np.array(model.predict(X_test))

    print(statistics(Y_test[:,1000:4000], pred_test[:,1000:4000]).round(4))
