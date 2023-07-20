import yfinance as yf
import pandas as pd
from django.shortcuts import render
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from datetime import datetime,timedelta,time
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,accuracy_score
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta,time
from django.shortcuts import render
import joblib
from django.http import HttpResponseRedirect
from django.urls import reverse


def _train_KNN(X_train, y_train, X_test, y_test):

    # Replace missing values with mean
    X_train = np.nan_to_num(X_train)
    y_train = np.nan_to_num(y_train)
    X_test = np.nan_to_num(X_test)
    y_test = np.nan_to_num(y_test)
    
    knn = KNeighborsRegressor()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 25)}
    
    # Use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    
    # Fit model to training data
    knn_gs.fit(X_train, y_train)
    
    # Save best model
    knn_best = knn_gs.best_estimator_
     
    # Check best n_neigbors value
    print(knn_gs.best_params_)
    
    prediction = knn_best.predict(X_test)

    print("KNN MSE:", mean_squared_error(y_test, prediction))
    
    return knn_best
    
    
def _train_random_forest(X_train, y_train, X_test, y_test):
    
    # Replace missing values with mean
    X_train = np.nan_to_num(X_train)
    y_train = np.nan_to_num(y_train)
    X_test = np.nan_to_num(X_test)
    y_test = np.nan_to_num(y_test)
    
    rf = RandomForestRegressor(random_state=42)
    
    # Create a dictionary of all values we want to test for n_estimators
    params_rf = {'n_estimators': [100, 300, 500, 800, 1000]}
    
    # Use gridsearch to test all values for n_estimators
    rf_gs = GridSearchCV(rf, params_rf, cv=5)
    
    # Fit model to training data
    rf_gs.fit(X_train, y_train)
    
    # Save best model
    rf_best = rf_gs.best_estimator_
    
    # Check best n_estimators value
    print(rf_gs.best_params_)
    
    prediction = rf_best.predict(X_test)

    print("Random Forest MSE:", mean_squared_error(y_test, prediction))
    
    return rf_best


def _ensemble_model(X_train, y_train, X_test, y_test):
    
    # Replace missing values with mean
    X_train = np.nan_to_num(X_train)
    y_train = np.nan_to_num(y_train)
    X_test = np.nan_to_num(X_test)
    y_test = np.nan_to_num(y_test)
    rf_model = _train_random_forest(X_train, y_train, X_test, y_test)
    knn_model = _train_KNN(X_train, y_train, X_test, y_test)
    
    # Create a dictionary of our models
    estimators=[('knn', knn_model), ('rf', rf_model)]
    
    # Create our voting regressor, inputting our models
    ensemble = VotingRegressor(estimators)
    
    #fit model to training data
    ensemble.fit(X_train, y_train)
    
    #test our model on the test data
    print("Ensemble Accuracy:", ensemble.score(X_test,y_test))
    
    return ensemble




import joblib


def predict(request):
    stocks = ['^GSPC', '^DJI', '^N225', '^IXIC', '^BSESN']
    stockName=['S and P 500','Dow Jones','Nikkei 500', 'NASDAQ', 'BSE Sensex']
    context = {'stocks': []}
    for i,stock in enumerate(stocks):
        # Load the ensemble model from disk, if it exists
        try:
            ensemble_model = joblib.load(f'ensemble_model_{stock}.joblib')
        except FileNotFoundError:
            ensemble_model = None

        df = yf.download(stock, period='1000d', interval='1d')
        df.reset_index(inplace=True)
        df.set_index("Date", inplace=True)
        df = df[['Close']]
        df.dropna(inplace=True)
        forecast_out = 30
        df['Prediction'] = df[['Close']].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = preprocessing.scale(X)
        X=np.round(X,4)
        X_forecast = X[-forecast_out:]
        X = X[:-forecast_out]

        y = np.array(df['Prediction'])
        y = y[:-forecast_out]
        y=np.round(y,4)

        # Train the models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        if ensemble_model is None:
            ensemble_model = _ensemble_model(X_train, y_train, X_test, y_test)
            joblib.dump(ensemble_model, f'ensemble_model_{stock}.joblib')

        predictions = ensemble_model.predict(X_forecast)
        predictions=np.round(predictions,4)
        
        print(stock)
        print("Ensemble Accuracy:", ensemble_model.score(X_test,y_test))
        y_pred = ensemble_model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)

        # Print the accuracy score in the console
        print("Ensemble model accuracy: {:.2f}%".format(accuracy * 100))

        stock_data = {'name': stockName[i], 'dates': [], 'prices': [], 'forecast': [], 'percent_change':[]}
        

        # Create a list of dates for the past 5 days
        today = datetime.today()
        past_dates = [(today - timedelta(days=x)).date() for x in range(5)]
        past_dates.reverse()
         # Reverse the order to display in ascending order

        # Add the past prices to the context
        for date in past_dates:
            try:
                stock_data['dates'].append(date.strftime('%Y-%m-%d'))
                price = np.round(df.loc[date]['Close'],4)
                stock_data['prices'].append(price)
            except KeyError:
                pass

        # Add the forecasted prices to the context
        for i, prediction in enumerate(predictions):
            stock_data['dates'].append((today + timedelta(days=i+1)).date().strftime('%Y-%m-%d'))
            stock_data['forecast'].append(prediction)
    
        latest_date = df.index.max()

# Add the latest price to the context, if it exists
        if latest_date in past_dates:
            latest_price = df.loc[latest_date]['Close']
            stock_data['latest_price'] = latest_price
            stock_data['percent_change'] = (stock_data['forecast'][-1] - latest_price) / latest_price * 100
        else:
            stock_data['latest_price'] = None
            stock_data['percent_change'] = None
        context['stocks'].append(stock_data)

    return render(request, 'predictions.html', context)

import os



    
def retrain_model(request):
    # Define the list of stocks
    stocks = ['^GSPC', '^DJI', '^N225', '^IXIC', '^BSESN']

    # Loop over each stock and delete its corresponding model file
    for stock in stocks:
        # Create the model file path
        file_path = f"ensemble_model_{stock}.joblib"
        
        # Check if the file exists and delete it
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Model deleted for {stock}")

    # Return the user to the predict page
    return HttpResponseRedirect(reverse('predict'))
