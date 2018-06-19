__author__ = 'Lupus'

#Acceptance criteria: Micro F1 > 0.8


import pandas as pandas
import numpy as np
from sklearn import *
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import sys


def run(training_set, test_set):

    td = pandas.read_csv(training_set, skiprows=0)

    #handling the NaN and empty cells
    td.replace('', np.nan, inplace=True)

    td = td.dropna(thresh=9)
    td = td.replace(['1. White', '2. Black', '3. Asian', '4. Other'], [1, 2, 3, 4])
    td = td.replace(['1. Never Married', '2. Married', '3. Widowed', '4. Divorced', '5. Separated'], [1, 2, 3, 4, 5])
    td = td.replace(['1. < HS Grad', '2. HS Grad', '3. Some College', '4. College Grad', '5. Advanced Degree'], [1, 2, 3, 4, 5])
    td = td.replace(['1. < HS Grad', '2. HS Grad', '3. Some College', '4. College Grad', '5. Advanced Degree'], [1, 2, 3, 4, 5])
    td = td.replace(['1. Industrial', '2. Information'], [1, 2])
    td = td.replace(['1. <=Good', '2. >=Very Good'], [1, 2])
    td = td.replace(['1. Yes', '2. No'], [1, 2])

    X_train = td.loc[:, td.columns != 'race']

    min_train = X_train.min()
    max_train = X_train.max()

    X_train =(X_train-min_train)/(max_train-min_train)

    y_train= td.loc[:, "race"]


    test_data = pandas.read_csv(test_set, skiprows=0)

    test_data.replace('', np.nan, inplace=True)
    test_data = test_data.dropna(thresh=9)
    test_data = test_data.replace(['1. White', '2. Black', '3. Asian', '4. Other'], [1, 2, 3, 4])
    test_data = test_data.replace(['1. Never Married', '2. Married', '3. Widowed', '4. Divorced', '5. Separated'], [1, 2, 3, 4, 5])
    test_data = test_data.replace(['1. < HS Grad', '2. HS Grad', '3. Some College', '4. College Grad', '5. Advanced Degree'], [1, 2, 3, 4, 5])
    test_data = test_data.replace(['1. < HS Grad', '2. HS Grad', '3. Some College', '4. College Grad', '5. Advanced Degree'], [1, 2, 3, 4, 5])
    test_data = test_data.replace(['1. Industrial', '2. Information'], [1, 2])
    test_data = test_data.replace(['1. <=Good', '2. >=Very Good'], [1, 2])
    test_data = test_data.replace(['1. Yes', '2. No'], [1, 2])

    X_test = test_data.loc[:, test_data.columns != 'race']

    X_test2 =(X_test-min_train)/(max_train-min_train)

    y_test = test_data.loc[:, "race"]


    #PCA I KLASIFIKACIJA
    pca = PCA(n_components=4)
    classifier = svm.SVC(kernel="linear", C= 1.0)

    pca.fit(X_train)
    x_transformed = pca.transform(X_train)
    y_transformed = pca.transform(X_test2)


    classifier.fit(x_transformed, y_train)

    predicted = classifier.predict(y_transformed)

    print(f1_score(y_test, predicted, average='micro') )
