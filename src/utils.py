import numpy as np
import pandas as pd
import joblib


def save_data(path='outputs/result.csv'):
    return pd.save(path)

def load_data(path='data/raw/creditcard.csv'):
    return pd.read_csv(path)

def get_X(df: pd.DataFrame):
    return df.drop('Class', axis=1)

def get_y(df: pd.DataFrame):
    return df['Class']

def save_model(model, model_name: str):
    joblib.dump(model, f'models/{model_name}.pkl')
    return print('Model saved.')

def load_model(model_name: str):
    return joblib.load(f'models/{model_name}.pkl')

def tune_threshold(thresholds: np.ndarray, precision, n=0.75):
    precision = precision[:-1]
    return thresholds[(precision >= n).argmax()]
