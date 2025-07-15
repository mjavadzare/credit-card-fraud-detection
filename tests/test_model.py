from src.predict import predict_lr_test, predict_lgbm_test
from src.train import X_test, y_test
from src.utils import save_predictions

def run():
    # threshold is optimized
    y_pred_lr = predict_lr_test(X=X_test)
    y_pred_lgbm = predict_lgbm_test(X=X_test)

    save_predictions(
        y_pred=y_pred_lr,
        y_true=y_test,
        algorithm='logistic_regression')
    print('prediction with Logistic Regression is now done. file saved.')
    save_predictions(
        y_pred=y_pred_lgbm,
        y_true=y_test, 
        algorithm='LGBM')
    print('prediction with LGBM is now done. file saved.')
    print('prediction is completed.')
    return y_pred_lr, y_pred_lgbm