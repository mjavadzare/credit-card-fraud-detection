import os

from src.train import (
    trained_lr_for_predict,
    trained_lgbm_for_predict,
    X_train,
    X_test,
)
from src.utils import save_model, load_model



# Save or Load Models
# LR
path_lr = 'models/Logistic-Regression.pkl'
if os.path.exists(path_lr):
    fitted_lr = load_model(model_name='Logistic-Regression')
    print('LR model loaded.')
else:
    fitted_lr = trained_lr_for_predict()
    save_model(fitted_lr, model_name='Logistic-Regression')

# LGBM
path_lgbm = 'models/LGBM.pkl'
if os.path.exists(path_lgbm):
    fitted_lgbm = load_model(model_name='LGBM')
    print('LGBM model loaded.')
else:
    fitted_lgbm = trained_lgbm_for_predict()
    save_model(fitted_lgbm, model_name='LGBM')




# For train data
def predict_lr_train():
    return fitted_lr.predict(X_train)

def predict_lgbm_train():
    return fitted_lgbm.predict(X_train)


optimized_threshold_lr = 5.342830925946151
optimized_threshold_lgbm = 0.7945137194363832



# For test data
def predict_lr_test(X=X_test):
    # seting new  opt threshold
    y_scores_test = fitted_lr.decision_function(X)
    y_pred_custom = (y_scores_test >= optimized_threshold_lr)
    return y_pred_custom

def predict_lgbm_test(X=X_test):
    # seting new  opt threshold
    y_scores_test = fitted_lgbm.predict_proba(X)[:, 1]
    y_pred_custom = (y_scores_test >= optimized_threshold_lgbm)
    return y_pred_custom
