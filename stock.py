"""
Pocinje obradjivanjem podataka koje ce obuhvatati proveru i eventualno uklanjanje celija i/ili kolona
koje ne sadrze odgovarajucu vrednost ili drasticno uticu na tacnost procene.
Nakon toga ce uslediti  pretvaranje kategorickih kolona, min-max normalizacija, uklanjanje outliera po potrebi.
Verovatno koriscenje PCA algoritma i konacno procena vrednosti koriscenjem neke od regresionih metoda.
"""

import pandas as pd
import numpy as np
import quandl, math
import datetime

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression,mutual_info_regression
from sklearn import preprocessing, model_selection, cross_decomposition
from sklearn import linear_model, svm
from sklearn.linear_model import *
from sklearn import metrics
import scipy as scipy
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def calculate_average_price(row):
    return (row['Adj. High'] + row['Adj. Low']) / 2

def calculate_change(row):
    return (row['Adj. Close'] - row['Adj. Open']) / row['Adj. Open']

def remove_outliers(data):
    temp_data = data;
  #  plt.plot(temp_data['Adj. Volume'])
  #  plt.show()

    indexes_to_drop = []
    for i in range(len(data.index)):
        if abs(temp_data['Adj. Volume'][i]) > 10000000 :
            indexes_to_drop.append(i)

    for i in range(len(data.index)):
        if abs(temp_data['change_percentage'][i]) > 0.03 :
            indexes_to_drop.append(i)

    temp_data.drop(temp_data.index[indexes_to_drop], inplace=True)

  #  plt.plot(temp_data['Adj. Volume'])
  #  plt.show()
    return temp_data

def calculate(company_codename, forecast_period):
    quandl.ApiConfig.api_key = "3dEBs4nQPtZwzGqB7XZu"

    #Loading company's stock prices from Quandl server
    company = "WIKI/" + company_codename
    df = quandl.get(company, start_date="2007-12-31", end_date="2017-12-31")

    #creating new column with average daily price which will be predicted
    #average_price column will be derived from Adj. High and Adj. Low columns
    df['average_price'] = df.apply(calculate_average_price, axis=1)
    df['change_percentage'] = df.apply(calculate_change, axis=1)

   #df = discard_columns(df)

   # print(df.describe())
    df = remove_outliers(df)

    #Creating new column that will represent average stock price in some future
    # this column will be created by shifting average_price value so that the new
    # column represents average_price value predefined time in future
    # -1 tell to shift backwards, or to move cell values up
    df['predicted_price'] = df['average_price'].shift(-forecast_period)
    #df = remove_nan_cells(df)
    df.dropna(inplace=True)

    #Creating X that will contain features to base regression on
    X = df.loc[:, df.columns != 'predicted_price']

    #creating y that will contain labels to compare predicted values with
    y = df.loc[:, "predicted_price"]
   # X.dropna(inplace=True)

    #for best features to be choosen
    test = SelectKBest(score_func=mutual_info_regression, k=5)
    X = test.fit_transform(X, y)
    # summarize scores
    #np.set_printoptions()

    #print(df.head())
    #Scaling X values
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)

    #Creating train and test frames, by the 80% of all being training and 20% being testing values
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    pca = PCA(n_components=5)
    pca.fit(X_train)
    x_transformed = pca.transform(X_train)
    y_transformed = pca.transform(X_test)


    #n_jobs was set to -1 in case of multiple calculations, in that case all CPUs would be used
    regressor = Lasso(tol=0.01, max_iter=10000)
    regressor.fit(x_transformed, y_train)

    y_pred = regressor.predict(y_transformed)
   # print("Prediction period in days: ", forecast_period)
   # print("Root Mean Squared Error for prediction of " + company_codename + " average prices: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
   # print("***************************************************************************************")
    df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    return df2

"""
Nisam znao da li moramo kao kod domaćih zadataka samo rezultat da ispišemo
pa sam sav kod koji plotuje i formatira ispis zakomentarisao
"""
def run() :
    google_dataframe = calculate("GOOGL", 15)
   # amazon_dataframe = calculate("AMZN", 30)
   # tesla_dataframe = calculate("TSLA", 60)

"""
    google_dataframe['Actual'].plot(figsize=(25, 11), color="green", label='Google Actual ')
    google_dataframe['Predicted'].plot(figsize=(25, 11), color="red", label='Google Predicted ')

    amazon_dataframe['Actual'].plot(figsize=(15, 6), color="blue", label='Amazon Actual ')
    amazon_dataframe['Predicted'].plot(figsize=(15, 6), color="yellow", label='Amazon Predicted ')
    tesla_dataframe['Actual'].plot(figsize=(15, 6), color="purple", label='Tesla Actual ')
    tesla_dataframe['Predicted'].plot(figsize=(15, 6), color="black", label='Tesla Predicted ')
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Average price in USD')
    plt.show()
"""


if __name__ == "__main__":
    run()