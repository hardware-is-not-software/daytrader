import json
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import logging

def ensure_stock_dir(stock):
    """Ensure the stock-specific results directory exists"""
    os.makedirs(f'results/{stock}', exist_ok=True)

def store_facts_to_file(facts, stock, filename='stock.json'):
    """Store facts in stock-specific directory"""
    ensure_stock_dir(stock)
    filepath = f'results/{stock}/{filename}'
    with open(filepath, 'w') as f:
        json.dump(facts, f, indent=4)

def save_to_csv(data, stock, filename=None):
    """Save data to CSV in stock-specific directory"""
    ensure_stock_dir(stock)
    if filename is None:
        filename = get_csv_filename(stock)
    filepath = f'results/{stock}/{filename}'
    data.to_csv(filepath)

def load_from_csv(stock, filename=None):
    """Load CSV from stock-specific directory"""
    if filename is None:
        filename = get_csv_filename(stock)
    filepath = f'results/{stock}/{filename}'
    if os.path.exists(filepath):
        return pd.read_csv(filepath, index_col=0, parse_dates=True, header=0)
    return None

def get_csv_filename(ticker):
    """Generate CSV filename for a given ticker"""
    return f"{ticker.upper()}.csv"

def get_daily_data(stock):
    # Use current date as end date
    end_date = datetime.now()
    # Go back 2 years for start date
    start_date = end_date - timedelta(days=730)
    
    # Format dates as strings in YYYY-MM-DD format for yfinance
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    try:
        ticker = yf.Ticker(stock)
        data = ticker.history(start=start_date_str, end=end_date_str, interval="1d")
        
        if len(data) == 0:
            logging.error(f"\nError: No data found for ticker '{stock}'")
            logging.error("Please verify the stock symbol is correct (e.g., AAPL, MSFT, GOOGL)")
            sys.exit(1)
        return data
    except Exception as e:
        logging.error(f"\nError: Failed to fetch data for ticker '{stock}'")
        logging.error("Please verify the stock symbol is correct (e.g., AAPL, MSFT, GOOGL)")
        logging.error(f"Error details: {str(e)}")
        sys.exit(1)

def validate_data(data, stock):
    """Validate that we have proper stock data"""
    if data is None or len(data) == 0:
        logging.error(f"\nError: No valid data found for ticker '{stock}'")
        logging.error("Please delete the CSV file and try again to fetch fresh data.")
        sys.exit(1)
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.error(f"\nError: Missing required columns in data: {missing_columns}")
        logging.error("Please delete the CSV file and try again to fetch fresh data.")
        sys.exit(1)
