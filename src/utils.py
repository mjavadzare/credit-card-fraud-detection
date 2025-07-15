import os
from datetime import datetime

import numpy as np
import pandas as pd
import joblib


def load_data(path='data/raw/creditcard.csv'):
    return pd.read_csv(path)

def get_X(df: pd.DataFrame):
    return df.drop('Class', axis=1)

def get_y(df: pd.DataFrame):
    return df['Class']

def save_model(model, model_name: str):
    joblib.dump(model, f'models/{model_name}.pkl')
    return print(f'{model_name} Model Saved.')

def load_model(model_name: str):
    return joblib.load(f'models/{model_name}.pkl')

def tune_threshold(thresholds: np.ndarray, precision, n=0.75):
    precision = precision[:-1]
    return thresholds[(precision >= n).argmax()]

def save_predictions(y_pred, y_proba=None, y_true=None, algorithm='prediction'):
    timestamp = datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    path = f'outputs/predictions/{algorithm}/{timestamp}.csv'
    # check if directory existed, if not, then make one
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {'predicted_label': y_pred,
            'predicted_proba' : y_proba if y_proba is not None else np.nan,
            'true_label' : y_true if y_true is not None else np.nan,
            }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

def save_confusion_matrix_error(labels, name, y_pred, y_true):
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        path = f'outputs/plots/confusion_matrix_{name}_errors.png'
        weights = (y_pred != y_true)
        display = ConfusionMatrixDisplay.from_predictions(
            y_pred=y_pred,
            y_true=y_true,
            sample_weight=weights,
            display_labels=labels,
            normalize='true',
            values_format='.3%')
        display.plot()
        plt.title(f'{name} Confusion Matrix Errors')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()