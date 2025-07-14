from .train import (
    fitted_lr_train,
    fitted_lgbm_train,
    X_train,
    X_test,
)
from .utils import save_model, load_model


'''Saved Models'''
# # Fitted models
# fitted_lr = fitted_lr_train()
# fitted_lgbm = fitted_lgbm_train()
# save_model(fitted_lr, model_name='Logistic-Regression')
# save_model(fitted_lgbm, model_name='LGBM')

fitted_lr = load_model('Logistic-Regression')
fitted_lgbm = load_model('LGBM')


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
