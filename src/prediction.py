import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from darts.models import BlockRNNModel

DATA_PATH = 'data/Fundamental+Techniqul Data/SET50_CLEAN_DATA_Version2/'
ID_PATH = 'data/Fundamental+Techniqul Data/ID_Name.csv'
RESULT_PATH = 'result/Prediction.csv'
FOCUS_COMPONENT = 'Close'
RETAIN_COMPONENTS = ["Open", "High", "Low", "PE", "PBV", "T_EPS", "FSCORE", "Vol",
                           "Buy Vol", "Sell Vol", "ATO/ATC", "EMA25", "EMA50", "EMA200", "MACD", "RSI"]
MODEL = BlockRNNModel(input_chunk_length= 5, output_chunk_length = 7, model='LSTM',n_epochs = 18,)
MAX_SPLIT_SIZE = 60
PREDICT_SIZE = 7


def preprocess_data(data,split):
    data = data.dropna()
    serie = data[FOCUS_COMPONENT]
    past_covariate = data[RETAIN_COMPONENTS].apply(pd.to_numeric, errors='coerce').fillna(method='ffill').fillna(method='bfill')
    serie_ts = TimeSeries.from_dataframe(serie.to_frame())
    past_cov_ts = TimeSeries.from_dataframe(past_covariate)
    scaler = StandardScaler()
    scaler_dataset = Scaler(scaler)
    scaled_serie_ts = scaler_dataset.fit_transform(serie_ts)
    if split == 0:
        training_scaled = scaled_serie_ts
    else:
        training_scaled = scaled_serie_ts[:-split]
        past_cov_ts = past_cov_ts[:-split]
    return training_scaled, past_cov_ts, scaler_dataset

def predict_next_n_days(training_scaled, past_cov_ts, scaler_dataset):
    """Predict next n days' closing prices for each stock."""
    MODEL.fit(training_scaled, past_covariates=past_cov_ts, verbose=False)
    forecast = MODEL.predict(PREDICT_SIZE, verbose=False)
    in_forecast = scaler_dataset.inverse_transform(forecast)
    return in_forecast

def generate_output(predictions, stock_data, id_name_map, split, stock):
    """Format the predictions into the desired output."""
    output = []
    predict_date = stock_data['Date'].iloc[-split-1]
    # Convert id_name_map to dictionary for faster lookups
    name_to_id = id_name_map.set_index('StockName')['Stock_ID'].to_dict()
    # Get stock_id using the dictionary; if not found, raise an error
    if stock not in name_to_id:
        raise ValueError(f"Stock name {stock} not found in id_name_map")
    stock_id = int(name_to_id[stock])
    
    for _, row in predictions.iterrows():
        date = row[0]
        pred_value = round(row[1], 2)
        output.append([predict_date, stock_id, date, pred_value])
            
    output_df = pd.DataFrame(output, columns=["Predict_Date", "Stock_ID", "Date", "Closing_Price"])
    return output_df.to_csv(RESULT_PATH, mode='a', header=not os.path.exists(RESULT_PATH))

def finalize_csv():
    final_df = pd.read_csv(RESULT_PATH)
    
    # Drop old index column if it exists
    if 'Unnamed: 0' in final_df.columns:
        final_df.drop('Unnamed: 0', axis=1, inplace=True)
    
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': 'Order_ID'}, inplace=True)
    final_df.to_csv(RESULT_PATH, index=False)


def main():
    file_list = sorted(os.listdir(DATA_PATH))
    if os.path.exists(RESULT_PATH):
        os.remove(RESULT_PATH)
    
    for split in range(MAX_SPLIT_SIZE, -1, -1):
        for stock_file in file_list:
            if not stock_file.endswith(".csv"):
                continue
        
            print(f"Predicting {stock_file} ...")
            # Load the data
            stock_data = pd.read_csv(os.path.join(DATA_PATH, stock_file))
            id_name_map = pd.read_csv(ID_PATH)

            # Preprocess data
            training_scaled, past_cov_ts, scaler_dataset = preprocess_data(stock_data, split)

            # Predict
            predictions = predict_next_n_days(training_scaled, past_cov_ts, scaler_dataset)

            # Generate output
            # Handle the date mapping for predictions
            # Handle the date mapping for predictions
            if split - PREDICT_SIZE <= 0:  # Modify this line to handle split == 0 as well
                last_known_date = pd.to_datetime(stock_data['Date'].iloc[-1], dayfirst=True)
                
                if split == 0:  # Add this condition
                    difference = PREDICT_SIZE
                    future_dates = [(last_known_date + pd.Timedelta(days=i+1)).strftime('%-d/%-m/%Y') for i in range(difference)]
                    date = future_dates
                else:
                    difference = -1 * (split - PREDICT_SIZE)
                    future_dates = [(last_known_date + pd.Timedelta(days=i+1)).strftime('%-d/%-m/%Y') for i in range(difference)]
                    date = stock_data['Date'][-split:].tolist() + future_dates
            else:
                date = stock_data['Date'][- split:len(stock_data) - split+PREDICT_SIZE].reset_index(drop=True)

            prediction_df = predictions.pd_dataframe().reset_index(drop=True)
            date_df = pd.DataFrame(date, columns=['Date'])
            combined_df = pd.concat([date_df, prediction_df], axis=1).reset_index(drop=True)

            generate_output(combined_df, stock_data, id_name_map, split, stock_file.split(".")[0])

if __name__ == "__main__":
    main()
    finalize_csv()