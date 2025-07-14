from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from..src.utils import load_data, get_X, get_y
from.data_preprocessing import all_transformation

df = load_data()

train, test = train_test_split(df,
                               test_size=0.2,
                               stratify=df['Class'],
                               random_state=10
                               )

X_train, X_test = get_X(train), get_X(test)
y_train, y_test = get_y(train), get_y(test)


# Logistic Regression
lr_pipeline = imbPipeline([
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
lgbm_pipeline = imbPipeline([
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


def fitted_lr_train():
    return lr_pipeline.fit(X_train, y_train)

def fitted_lgbm_train():
    return lgbm_pipeline.fit(X_train, y_train)
