from .train import (
    lr_pipeline,
    lgbm_pipeline,
    X_train,
    y_train
)
from .utils import tune_threshold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve


# Finding Optimized threshold for Logistic Regression
y_scores_lr = cross_val_predict(
    lr_pipeline,
    X_train,
    y_train,
    cv=5,
    method='decision_function',
    n_jobs=-1
)

precision_lr, recall_lr, thresholds_lr = precision_recall_curve(
    y_true=y_train,
    y_score=y_scores_lr,
)

optimized_threshold_lr = tune_threshold(
    thresholds=thresholds_lr,
    precision=precision_lr,
    n=0.75
)
# optimized_threshold_lr = 5.342830925946151




# Finding Optimized threshold for LGBM
y_scores_lgbm = cross_val_predict(
    lgbm_pipeline,
    X_train,
    y_train,
    cv=5,
    method='predict_proba',
    n_jobs=-1
)

y_scores_lgbm_fraud = y_scores_lgbm[:, 1]  # probability of being a fraud

precision_lgbm, recall_lgbm, thresholds_lgbm = precision_recall_curve(
    y_true=y_train,
    y_score=y_scores_lgbm_fraud,
)

optimized_threshold_lgbm = tune_threshold(
    thresholds=thresholds_lgbm,
    precision=precision_lgbm,
    n=0.90
)

# optimized_threshold_lr = 0.7945137194363832