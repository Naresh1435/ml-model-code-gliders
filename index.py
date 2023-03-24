from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_selection import SelectKBest, f_regression

app = FastAPI()

def BasicModel():
    images = list()
    def FigureToImg(obj):
        img_bytes = io.BytesIO()
        obj.savefig(img_bytes,format='png')
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode()
        images.append(img_base64)

    df = pd.read_csv('/Users/naresh_dev/Developments/SMART ANALYTICS/flask_app/new_sales-2.csv')
    df__parameters = pd.read_csv('/Users/naresh_dev/Developments/SMART ANALYTICS/flask_app/sales.csv')
    df__parameters.dropna(inplace=True)
    df.dropna(inplace=True)
    X_train, X_test, y_train, y_test = train_test_split( df.drop('Revenue', axis=1), df['Revenue'], test_size=0.2)
# Select the top 5 features based on F-test score
    selector = SelectKBest(score_func=f_regression, k=5)
    selector.fit(X_train, y_train)

# Print the names of the selected features
    selected_features = X_train.columns[selector.get_support()]
    
    print(f'Selected features: {selected_features}')

    selected_featured_df = df[selected_features]
    print(selected_featured_df.info())
    X_train, X_test, y_train, y_test = train_test_split( selected_featured_df, df['Revenue'], test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    print(f'R^2 score: {score:.2f}')
    # Group the data by quarter and product line, and calculate the total revenue for each group
    grouped = df__parameters.groupby(['Quarter', 'Product line']).agg({'Revenue': 'sum'})

    # Pivot the data to create a table with quarters as columns and product lines as rows
    pivoted = grouped.pivot_table(index='Product line', columns='Quarter', values='Revenue')

    # Plot the results as a stacked bar chart
    pivoted.plot(kind='bar', stacked=True, figsize=(8, 6))
    plt.title('Quarterly Revenue by Product Line')
    plt.xlabel('Product Line')
    plt.ylabel('Revenue')
    FigureToImg(plt)
    # Group the data by retailer country and year, and calculate the total revenue for each group
    grouped1 = df__parameters.groupby(['Retailer country', 'Year']).agg({'Revenue': 'sum'})
    # Calculate the year-over-year growth rate for each country
    grouped1['Growth'] = grouped1['Revenue'].pct_change()

    # Plot the results as a line chart
    for country in grouped1.index.get_level_values(0).unique():
        data = grouped1.loc[country]
        plt.plot(data.index.get_level_values(0), data['Growth'], label=country)

    plt.title('Year-over-Year Revenue Growth by Retailer Country')
    plt.xlabel('Year')
    plt.ylabel('Growth Rate')
    plt.legend()
    FigureToImg(plt)
    grouped = df__parameters.groupby('Product').agg({'Revenue': 'sum'})
    top_products = grouped.sort_values('Revenue', ascending=False).head(10)
    top_products.plot(kind='barh', figsize=(8, 6))
    plt.title('Top 10 Products by Revenue')
    plt.xlabel('Revenue')
    plt.ylabel('Product')
    FigureToImg(plt)




    return {'info' : list(selected_features), 'score' : score, 'images' : images}

def Arimaforcasting():
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
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes,format='png')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode()
    return {'TimeSeriesGraph' : img_base64 }


@app.get('/')
async def home():
    basicModel = BasicModel()
    forecast = Arimaforcasting() 
    return ({'resA': basicModel, 'resB' : forecast })

if __name__ == '__main__':
    app.run()