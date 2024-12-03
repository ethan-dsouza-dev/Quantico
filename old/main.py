import numpy as np
import pandas as pd
from datetime import datetime
from data import get_indicator_data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score

tickers = ["NVDA", "TSLA", "AMZN", "AAPL", "MSFT", "AMD", "META", "GOOGL", "NFLX"]
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
model_summaries = []

for ticker in tickers:
    df = get_indicator_data(ticker)

    # Selecting our Predictor Variables from the df
    X = df.iloc[:,:9]

    # Making a df where 1 means tomorrows closing price is higher than todays,
    # -1 means tomorrows closing price is lower than todays
    y = np.where(df['Close'].shift(-1) > df['Close'],1,-1)

    # splitting dataset into training and test data
    split = int(0.7*len(df))
    training_X, testing_X, training_Y, testing_Y = X[:split], X[split:], y[:split], y[split:]

    # fitting LR model
    model = LogisticRegression()
    model = model.fit (training_X, training_Y)

    model_summary = pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
    model_summaries.append(model_summary)

    start_date = '2024-10-01'
    end_date = datetime.today().strftime('%Y-%m-%d') # sets end date to todays date

    # Create a list containing the start and end dates
    prediction_date = [start_date, end_date]

    # Use the list of dates for prediction
    probability = model.predict_proba(testing_X)

    # Testing model with our testing data
    predicted = model.predict(testing_X)

    # Calculate model metrics and store them
    accuracy_scores.append(metrics.accuracy_score(testing_Y, predicted))
    precision_scores.append(metrics.precision_score(testing_Y, predicted, pos_label=1))
    recall_scores.append(metrics.recall_score(testing_Y, predicted, pos_label=1))
    f1_scores.append(metrics.f1_score(testing_Y, predicted, pos_label=1))


    # Finding which dates that it predicted to buy
    # Find the dates with predicted buy signals (Signal = 1)
    predicted_labels = 1
    buy_signal_dates = prediction_date[predicted_labels == 1]

    # Print the date(s) with buy signal(s)
    # print("\nDate(s) with Buy Signal(s):")
    # print(buy_signal_dates)

# Calculate and print average statistics across all tickers
print("\nAverage Model Statistics across all tickers:")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
