import pandas as pd

# Load dataset
csv_path = 'datasets/us_stock_2020_to_2024.csv'
csv_path_out = 'datasets/all_stocks.csv'

df=pd.read_csv(csv_path)

df = df[['Date Time','S&P_500_Price', 'Nasdaq_100_Price', 'Apple_Price', 'Meta_Price','Tesla_Price','Google_Price','Nvidia_Price','Amazon_Price', 'Microsoft_Price','Berkshire_Price','Netflix_Price','Amazon_Price']]
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d-%m-%Y')
df = df.sort_values(by='Date Time')

df.to_csv(csv_path_out)