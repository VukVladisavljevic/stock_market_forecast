"""
Pocinje obradjivanjem podataka koje ce obuhvatati proveru i eventualno uklanjanje celija i/ili kolona
koje ne sadrze odgovarajucu vrednost ili drasticno uticu na tacnost procene.
Nakon toga ce uslediti  pretvaranje kategorickih kolona, min-max normalizacija, uklanjanje outliera po potrebi.
Verovatno koriscenje PCA algoritma i konacno procena vrednosti koriscenjem neke od regresionih metoda.


"""

# import relevant modules
import pandas as pd
import numpy as np
import quandl, math
import datetime

# Machine Learning
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#Visualization
import matplotlib.pyplot as plt

min_vrednost = -1
max_vrednost = -1

min_vrednost_y = -1
max_vrednost_y = -1

def normalize(x):
    global min_vrednost
    global max_vrednost
    ret_x = list()
    min_vrednost = x.min()
    max_vrednost = x.max()
    for i in range(0, len(x)):
        val1 = x[i]-x.min()
        val2 = x.max()-x.min()
        ret_x.append(val1/val2)
    return ret_x

def normalize_y(x):
    global min_vrednost_y
    global max_vrednost_y
    ret_x = list()
    min_vrednost_y = x.min()
    max_vrednost = x.max()
    for i in range(0, len(x)):
        val1 = x[i]-x.min()
        val2 = x.max()-x.min()
        ret_x.append(val1/val2)
    return ret_x

def normalize_test_y(x):
    global min_vrednost_y
    global max_vrednost_y
    ret_x = list()
    for i in range(0, len(x)):
        val1 = x[i]-min_vrednost_y
        val2 = max_vrednost_y-min_vrednost_y
        ret_x.append(val1/val2)
    return ret_x

def normalize_test(x):
    global min_vrednost
    global max_vrednost
    ret_x = list()
    for i in range(0, len(x)):
        val1 = x[i]-min_vrednost
        val2 = max_vrednost-min_vrednost
        ret_x.append(val1/val2)
    return ret_x


def denormalize(x):
    global min_vrednost_y
    global max_vrednost
    denormalized_values = list()
    for i in range(0, len(x)):
        val = x[i]*(max_vrednost_y - min_vrednost_y) + min_vrednost_y
        denormalized_values.append(val)
    return denormalized_values


def calculate_rmse(x, y, b, m):
    error_sum = 0
    predicted_y_values = list()
    for i in range(0, len(x)):
        predicted_y_values.append(m*x[i] + b)

    denorm_y_predicted = denormalize(predicted_y_values)
    for i in range(0, len(x)):
        prediction_error = denorm_y_predicted[i] - y[i]
        error_sum += (prediction_error ** 2)

    mean_error = error_sum / float(len(x))
    return math.sqrt(mean_error)


def remove_outliers(x, y):
    x_list = list()
    y_list = list()

    for i in range(0, len(x)):
        if(x[i] < 3050 and y[i] > 1200):
            continue

        if(x[i]> 4100  and y[i] < 1350):
            continue
        x_list.append(x[i])
        y_list.append(y[i])


    return np.asarray(x_list), np.asarray(y_list)

def gradient_descent_function(x1, y1, alpha_value, num_iterations):
    b = -1
    m = -1
    size = len(x1)

    for i in range(num_iterations):
        for j in range(0, len(x1)):
            x = x1[j]
            y = y1[j]
            b -= alpha_value*(-(2/size) * (y - ((m * x) + b)))
            m -= alpha_value*(-(2/size) * x * (y - ((m * x) + b)))
    return b, m

def discard_columns(data):

    "Removing columns that aren't needed for this forecast"
    columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio', 'Adj. Volume','Adj. High', 'Adj. Low']
    data.drop(columns, inplace=True, axis=1)
    return data

def calculate_average_price(row):
    return (row['Adj. High'] + row['Adj. Low']) / 2

def run():
    df = quandl.get("WIKI/AMZN")

    "Calculate average price for each day"
    df['average_price'] = df.apply(calculate_average_price, axis=1)

    "discard columns"
    df = discard_columns(df)

    # Visualization

    df['average_price'].plot(figsize=(15, 6), color="green")
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
  #  plt.show()


    # pick a forecast column
    forecast_col = 'average_price'

    # Chosing 30 days as number of forecast days
    forecast_out = int(30)

    # Creating label by shifting 'Adj. Close' according to 'forecast_out'
    df['prediction'] = df[forecast_col].shift(-forecast_out)

    # Define features Matrix X by excluding the label column which we just created
    X = np.array(df.drop(['prediction'], 1))

    # Using a feature in sklearn, preposessing to scale features
    X = normalize(X)
   # print(X)

    # X contains last 'n= forecast_out' rows for which we don't have label data
    # Put those rows in different Matrix X_forecast_out by X_forecast_out = X[end-forecast_out:end]

    X_forecast_out = X[-forecast_out:]
    X = X[:-forecast_out]

    # Similarly Define Label vector y for the data we have prediction for
    # A good test is to make sure length of X and y are identical
    y = np.array(df['prediction'])
    y = y[:-forecast_out]

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    # Test
    accuracy = clf.score(X_test, y_test)
    print("Accuracy of Linear Regression: ", accuracy)


if __name__ == "__main__":
    run()