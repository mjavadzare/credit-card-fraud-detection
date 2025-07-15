from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from src.utils import load_data, get_X, get_y
from src.data_preprocessing import all_transformation


df = load_data()

train, test = train_test_split(df,
                               test_size=0.2,
                               stratify=df['Class'],
                               random_state=10
                               )

X_train, X_test = get_X(train), get_X(test)
y_train, y_test = get_y(train), get_y(test)


# Logistic Regression
lr_train_pipeline = imbPipeline([
    ('transformation', all_transformation),
    ('smote', SMOTE(random_state=10)),
    # elasticnet showed better performance in terms of runtime.
    ('logistic_r', LogisticRegression(
            penalty='elasticnet', 
            l1_ratio=0.8,
            solver='saga',
            random_state=10,
            max_iter=1000,
            n_jobs=-1)
    )
])


# LGBM (better performance)
lgbm_train_pipeline = imbPipeline([
    ('transformation', all_transformation),
    ('smote', SMOTE(random_state=10)),
    ('lgbm', LGBMClassifier(
        boosting_type='gbdt',
        learning_rate=0.2,
        num_leaves=70,
        subsample=0.5,
        data_sample_strategy='goss',
        random_state=10,
        n_jobs=-1,
        verbose=-1
    ))
])

# LR
# Train Model with SMOTE
lr_train_pipeline.fit(X_train, y_train)
fitted_transformation_lr = lr_train_pipeline.named_steps['transformation']
fitted_lr = lr_train_pipeline.named_steps['logistic_r']

# Trained model in a pipeline without SMOTE
lr_predict_pipeline = Pipeline([
    ('transformation', fitted_transformation_lr),
    ('logistic_r', fitted_lr)
])

# LGBM
# Train Model with SMOTE
lgbm_train_pipeline.fit(X_train, y_train)
fitted_transformation_lgbm = lgbm_train_pipeline.named_steps['transformation']
fitted_lgbm = lgbm_train_pipeline.named_steps['lgbm']

# Trained model in a pipeline without SMOTE
lgbm_predict_pipeline = Pipeline([
    ('transformation', fitted_transformation_lgbm),
    ('lgbm', fitted_lgbm)
])


def trained_lr_for_predict():
    return lr_predict_pipeline

def trained_lgbm_for_predict():
    return lgbm_predict_pipeline