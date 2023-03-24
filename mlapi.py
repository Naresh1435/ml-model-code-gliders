#Bring in the modules
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/naresh_dev/Developments/SMART ANALYTICS/flask_app/new_sales-2.csv')
df.dropna(inplace=True)
print(df.columns)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Revenue', axis=1), df['Revenue'], test_size=0.2, random_state=42)

# Fit a linear regression model to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing data
score = model.score(X_test, y_test)
print(f'R^2 score: {score:.2f}')



from sklearn.feature_selection import SelectKBest, f_regression

# Select the top 5 features based on F-test score
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X_train, y_train)

# Print the names of the selected features
selected_features = X_train.columns[selector.get_support()]
print(f'Selected features: {selected_features}')

df1 = pd.read_csv('/Users/naresh_dev/Developments/SMART ANALYTICS/flask_app/sales.csv')
df1.dropna(inplace=True)

arima_df = df1
arima_df['Quarter'] = arima_df['Quarter'].str.replace(r'\b\d{4}\b','')
arima_df['Quarter'] = arima_df['Quarter'].str.strip().str.replace('Q', '')

arima_df['date'] = pd.to_datetime(arima_df['Year'].astype(str) +'Q'+ arima_df['Quarter'].astype(str))
arima_df.set_index('date',inplace=True)
grouped_df = arima_df.groupby(pd.Grouper(freq='Q')).agg({'Revenue': 'sum'})

arima_df.drop(['Year','Quarter'],axis=1,inplace=True)






# Fit an ARIMA model to the time series
model = ARIMA(grouped_df, order=(2, 1, 1))
result = model.fit()

# Print the summary of the model
forecast = result.forecast(steps=8)
plt.figure(figsize=(8,6))
plt.plot(grouped_df.index, grouped_df['Revenue']  ,label='Orginal Data')
plt.plot(forecast.index, forecast ,label='ARIMA Forecast')
plt.title('Arima Forecast for Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.show()
# Print the forecasted values
