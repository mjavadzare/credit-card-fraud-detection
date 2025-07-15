import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer,
    QuantileTransformer
)
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA


default_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
)

Time_rbf_transformer = FunctionTransformer(
    rbf_kernel,
    feature_names_out='one-to-one',
    kw_args={
        'Y':[[79000]],
        'gamma':1e-8
    }
)

q_heavy_tail_pipeline = make_pipeline(
    QuantileTransformer(output_distribution='normal'),
    StandardScaler()
)

# V2, V5, V7 with Amount
V2_amount_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2)
)

V5_amount_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2)
)

V7_amount_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=2)
)

heavy_tail_features = [
    'V1', 'V2', 'V3', 'V4', 'V5',
    'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V14', 'V15',
    'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25',
    'V27', 'V28', 'Amount'
]

all_transformation = ColumnTransformer(
    transformers=[
        ('Time_rbf', Time_rbf_transformer, ['Time']),
        ('heavy_tail', q_heavy_tail_pipeline, heavy_tail_features),
        ('V2_amount', V2_amount_pipeline, ['V2', 'Amount']),
        ('V5_amount', V5_amount_pipeline, ['V5', 'Amount']),
        ('V7_amount', V7_amount_pipeline, ['V7', 'Amount']),
    ],
    remainder=default_pipeline,
    force_int_remainder_cols=False
)


def preprocess(df: pd.DataFrame):
    return all_transformation.fit_transform()