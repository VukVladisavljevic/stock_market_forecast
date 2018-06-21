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

from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import *
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def discard_columns(data):

    #Removing columns that aren't needed for this forecast
    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend',
               'Split Ratio','Adj. High', 'Adj. Low','Adj. Open']
    data.drop(columns, inplace=True, axis=1)
    return data

def calculate_average_price(row):
    return (row['Adj. High'] + row['Adj. Low']) / 2

def calculate_change(row):
    return (row['Adj. Close'] - row['Adj. Open']) / row['Adj. Open']

def remove_nan_cells(data):

    #handling the NaN and empty cells
    data.replace('', np.nan, inplace=True)
    data = data.dropna(thresh=5)

    return data


def remove_outliers(data):
    temp_data = data;
    plt.plot(temp_data['Adj. Volume'])
    plt.show()

    indexes_to_drop = []
    for i in range(len(data.index)):
        if abs(temp_data['Adj. Volume'][i]) > 10000000 :
            indexes_to_drop.append(i)

    for i in range(len(data.index)):
        if abs(temp_data['change_percentage'][i]) > 0.03 :
            indexes_to_drop.append(i)

    temp_data.drop(temp_data.index[indexes_to_drop], inplace=True)

    plt.plot(temp_data['Adj. Volume'])
  #  plt.show()
    return temp_data

def run():
    quandl.ApiConfig.api_key = "3dEBs4nQPtZwzGqB7XZu"
    #Loading Amazon\Google stock prices from Quandl server
    df = quandl.get("WIKI/GOOGL", start_date="2012-12-31", end_date="2017-12-31")
    print(df.head())
    #creating new column with average daily price which will be predicted
    #average_price column will be derived from Adj. High and Adj. Low columns
    df['average_price'] = df.apply(calculate_average_price, axis=1)
    df['change_percentage'] = df.apply(calculate_change, axis=1)

    #df = discard_columns(df)


    #df = remove_outliers(df)


    #Number of days for forecasting
    forecast_period = 30

    #Creating new column that will represent stock price in some future
    #tj. predstavlja kolonu sa kojom cemo porediti vrednosti koje ce prognozirati algoritam
    df['predicted_price'] = df['Adj. Close'].shift(-forecast_period)
    df = remove_nan_cells(df)
    #df.dropna(inplace=True)

    X = df.loc[:, df.columns != 'predicted_price']
    #ovo ispod pravi kao listu a gore je bas dataframe
    #X = np.array(df.drop(['predicted_price'], 1))

    y = df.loc[:, "predicted_price"]

    #Normalizacija/Skaliranje/Standardizacija
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)



    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1,  random_state=142)


    regressor = Lasso()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df2)

    # Showing stock value over time
    df['average_price'].plot(figsize=(15, 6), color="green")
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
   # plt.show()



    """
    X_test2 =(X_test-X_train.min())/(X_train.max()-X_train.min())


    #PCA I KLASIFIKACIJA
    pca = PCA(n_components=3)
    regressor2 = svm.LinearSVR()
    regressor2.fit(X_train, y_train)

    pca.fit(X_train)
    x_transformed = pca.transform(X_train)
    y_transformed = pca.transform(X_test2)

    regressor2.fit(x_transformed, y_train)

    predicted2 = regressor2.predict(y_transformed)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted2)))
"""
if __name__ == "__main__":
    run()