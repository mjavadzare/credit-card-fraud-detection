from ..src.predict import predict_lr_test, predict_lgbm_test
from ..src.train import X_test


# threshold is optimized
y_pred_lr = predict_lr_test(X=X_test)
y_pred_lgbm = predict_lgbm_test(X=X_test)
