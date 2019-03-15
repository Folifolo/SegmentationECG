This project is part of the [ECG Segmentation by Neural Networks: Errors and Correction](https://arxiv.org/abs/1812.10386) article and is used to create, train and use neural network ensembles for ECG annotation.

REQUIREMENTS
------------
Packages are required to use the project: tensorflow, keras and BaselineWanderRemoval

A [LUDB](http://www.cyberheart.unn.ru/database) dataset was used for the experiments. We used [json file](https://drive.google.com/file/d/1LGXwTUIO4vDfocK4qT03B9acnOpWbAaU/view?usp=sharing)

QUICK START
-----------
In the beginning, you need to specify in utils.py the path to the folder with the data and working directories

To train members of the ensemble you need to run train_models.py. In result of his work in the folder specified in path_to_ensemble_models files with trained models will be stored. In path_to_ensemble_data stored data on which models were trained.

To use trained models for annotation, use the ensemble_predict function from ensemble/make_ensemble.py. In ensemble/visualisation.py there are functions to visualize the result, for example, draw_one plot the network prediction and ground thruth with the ECG.

To create a judge network trained on the members of the ensemble, you need to use the function train_judge from ensemble/judge.py. To obtain a annotation with the use of the judicial network, use the function ensemble_predict_with_judge. Usage example in main judge.py

To evaluate the results, you can use the function named statistics from metrics.py, which creates a pandas table with sensitivity, PPV, mean, and error variance values. Also there is a graphical method of evaluate the result in a file ensemble.visualisation.py: ranging. Ranging is plotting scatterogram of F1 score values at each of the patients.
