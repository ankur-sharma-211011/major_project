import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import os

CACHE_DIR = "price_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def download_stock_data(tickers, start_date, end_date):

    # Normalize dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # ==========================================
    # MULTI-TICKER CASE
    # ==========================================
    if isinstance(tickers, (list, tuple)):
        dfs = []

        for ticker in tickers:
            df = download_stock_data(ticker, start_date, end_date)
            df = df.rename(columns={"Adj Close": f"{ticker}_Adj Close"})
            dfs.append(df)

        return pd.concat(dfs, axis=1)

    # ==========================================
    # SINGLE TICKER CASE
    # ==========================================
    ticker = tickers
    file_path = os.path.join(CACHE_DIR, f"{ticker}.csv")

    df = pd.DataFrame()

    # ------------------------------------------
    # LOAD CACHE (with corruption handling)
    # ------------------------------------------
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0)

            # Remove corrupted rows
            df = df[~df.index.astype(str).str.contains("Ticker|Date|Price", na=False)]

            # Convert index safely
            df.index = pd.to_datetime(df.index, errors="coerce")

            # Drop invalid rows
            df = df[~df.index.isna()]

            # Keep only numeric
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna()

        except Exception:
            print(f"{ticker} cache corrupted → resetting")
            os.remove(file_path)
            df = pd.DataFrame()

    # ------------------------------------------
    # DETERMINE MISSING RANGE
    # ------------------------------------------
    need_download = False

    if df.empty:
        need_download = True
        dl_start = start_date
        dl_end = end_date

    else:
        cached_start = df.index.min()
        cached_end = df.index.max()

        if start_date < cached_start:
            need_download = True
            dl_start = start_date
            dl_end = cached_start - pd.Timedelta(days=1)

        elif end_date > cached_end:
            need_download = True
            dl_start = cached_end + pd.Timedelta(days=1)
            dl_end = end_date

    # ------------------------------------------
    # DOWNLOAD MISSING DATA
    # ------------------------------------------
    if need_download:
        try:
            new_data = yf.download(
                ticker,
                start=dl_start,
                end=dl_end,
                progress=False,
                auto_adjust=False,
            )

            if not new_data.empty and "Adj Close" in new_data:
                new_data = new_data[["Adj Close"]]

                df = pd.concat([df, new_data])
                df = df[~df.index.duplicated(keep="last")]
                df.sort_index(inplace=True)

                # 🔒 SAVE CLEAN FORMAT ONLY
                clean_df = df.copy()
                clean_df = clean_df[["Adj Close"]]
                clean_df.index.name = "Date"

                clean_df.to_csv(file_path)

        except Exception as e:
            print(f"{ticker} download failed: {e}")

    # ------------------------------------------
    # FINAL SLICE
    # ------------------------------------------
    df = df.loc[start_date:end_date]

    if df.empty:
        raise ValueError(f"No data available for {ticker}")

    # ------------------------------------------
    # ENSURE ORIGINAL FORMAT
    # ------------------------------------------
    df = df.copy()
    df.columns = ["Adj Close"]

    return df


def create_additional_features(stock_data):
    """
    Creates additional features for the stock data, such as moving averages
    :param stock_data: Dataset to create features for
    :return: pandas Dataframe with columns for moving averages
    """
    df = pd.DataFrame(stock_data)
    print(df.columns)
    df['20d_rolling_avg'] = stock_data.rolling(window=20).mean()
    df['40d_rolling_avg'] = stock_data.rolling(window=40).mean()
    
    # Add more features as needed

    df['10d_forward_return'] = df['Adj Close'].shift(-10) / df['Adj Close'] - 1
    df['5d_forward_return'] = df['Adj Close'].shift(-5) / df['Adj Close'] - 1
    df['30d_volatility'] = df['Adj Close'].pct_change().rolling(window=30).std()

    rolling_mean_40 = df['Adj Close'].rolling(window=40).mean()
    rolling_std_40 = df['Adj Close'].rolling(window=40).std()
    df['Bollinger_Band_Width'] = (2 * rolling_std_40) / rolling_mean_40
    
    df['30d_volatility'] = df['Adj Close'].pct_change().rolling(window=30).std()
    df['30d_volatility_lag_1'] = df['30d_volatility'].shift(1)

    df['drawdown'] = df['Adj Close'] / df['Adj Close'].rolling(window=40).max() - 1

    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
    
    df.drop(['30d_volatility'], axis=1, inplace=True)
    print(df.columns)
    return df


def prepare_data_for_ml(stock_data, lag_days=20):
    """
    Prepares the data for machine learning by creating lagged features
    :param stock_data: pandas Series or DataFrame
    :param lag_days: the number of days to lag the feature
    :return: pandas DataFrame with the original data and additional columns for each lagged feature
    """
    if isinstance(stock_data, pd.Series):
        df = pd.DataFrame(stock_data, columns=['Adj Close'])
    else:
        df = stock_data.copy()

    target_column = 'Adj Close' if 'Adj Close' in df.columns else df.columns[0]

    # Create lagged features based on the target column
    for i in range(1, lag_days + 1):
        df[f'lag_{i}'] = df[target_column].shift(i)

    df.dropna(inplace=True)
    return df


def train_model(model, X_train, y_train):
    """
    Trains the given model
    :param model: model to train
    :param X_train: features to train the model off of
    :param y_train: target values corresponding to X_train
    :return: trained model
    """
    model.fit(X_train, y_train)
    return model


def get_model_confidence(model, X_test, y_test):
    """
    Calculates the confidence in the model based on its performance.
    :param model: Trained machine learning model.
    :param DataFrame X_test: Test features.
    :param Series y_test: True values for the test set.
    :return: A confidence score for the model.
    """
    # Using the model's R-squared value as confidence
    r_squared = model.score(X_test, y_test)
    return r_squared


def predict_future_returns(model, stock_data):
    """
    Predicts future returns using the provided model and stock data.
    :param model: Trained machine learning model.
    :param stock_data: Data used for prediction (DataFrame or NumPy array).
    :return: Predicted future return.
    """
    # Check if stock_data is a DataFrame and prepare it if so
    if isinstance(stock_data, pd.DataFrame):
        prepared_data = prepare_data_for_ml(stock_data)
        features = prepared_data.drop('Adj Close', axis=1)
    else:
        # If stock_data is already a NumPy array, use it directly
        features = stock_data

    # Standardize features (if required by your model)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Predict using the model
    predictions = model.predict(features_scaled)

    # Here, you might want to aggregate the predictions in a specific way.
    # For simplicity, let's return the last prediction
    return predictions[-1]


def generate_investor_views(ticker, start_date, end_date, model_type='XGBoost', forward_days=20):
    """
    Generates future stock return predictions and model confidence for a given stock ticker within a specified date range, using a selected machine learning model.
    :param str ticker: ticker to generate investor views for
    :param start_date: start date for training in form 'YYYY-MM-DD'
    :param end_date: end date for training in form 'YYYY-MM-DD'
    :param model_type: type of machine learning model 
                       ('Linear Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost')
    :return: tuple with predicted returns and model's confidence
    """
    stock_data = download_stock_data(ticker, start_date, end_date)
    ml_stock_data_with_features = create_additional_features(stock_data)

    X = ml_stock_data_with_features.drop('Adj Close', axis=1)
    y = ml_stock_data_with_features['Adj Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle NaN values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Select and train the model
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    else:
        print("Please choose a valid model and try again!")
        return None

    trained_model = train_model(model, X_train, y_train)
    predicted_return = predict_future_returns(trained_model, X_test)
    confidence = get_model_confidence(trained_model, X_test, y_test)

    return predicted_return, confidence
