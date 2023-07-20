import yfinance as yf
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
from datetime import timedelta, datetime


def predict(request):
    stocks = ['AAPL', 'GOOG', 'AMZN', 'PLTR', 'TSLA']
    context = {'stocks': []}
    
    for stock in stocks:
        df = yf.download(stock, period='100d', interval='1d')
        df.reset_index(inplace=True)
        df.set_index("Date", inplace=True)
        df = df[['Close']]
        df.dropna(inplace=True)
        forecast_out = 30
        df['Prediction'] = df[['Close']].shift(-forecast_out)

        X = np.array(df.drop(['Prediction'], 1))
        X = preprocessing.scale(X)
        X_forecast = X[-forecast_out:]
        X = X[:-forecast_out]

        y = np.array(df['Prediction'])
        y = y[:-forecast_out]

        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Get the predictions
        predictions = regressor.predict(X_forecast)

        stock_data = {'name': stock, 'dates': [], 'prices': [], 'forecast': []}

        # Create a list of dates for the past 5 days
        today = datetime.today()
        past_dates = [(today - timedelta(days=x)).date() for x in range(5)]

        # Add the past prices to the context
        for date in past_dates:
            try:
                stock_data['dates'].append(date.strftime('%Y-%m-%d'))
                price = df.loc[date]['Close']
                stock_data['prices'].append(price)
            except KeyError:
                pass

        # Add the forecasted prices to the context
        for i, prediction in enumerate(predictions):
            stock_data['dates'].append((today + timedelta(days=i)).date().strftime('%Y-%m-%d'))
            stock_data['forecast'].append(prediction)

        context['stocks'].append(stock_data)

    return render(request, 'predictions.html', context)
