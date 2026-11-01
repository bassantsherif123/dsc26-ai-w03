"""Failure-risk baseline for Tawseel COD parcels."""
from __future__ import annotations

import pandas as pd
import math
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

TARGET = "failed"

# The raw columns of make_parcels() that the model may use at decision time (when the parcel
# is created). Nothing known only after the delivery attempt, not the label, not the row id.
# TODO: this is every column of the table. Keep only the ones known at decision time.
# FEATURE_COLUMNS: list[str] = [
#     "parcel_id", "customer_id", "merchant_id", "area", "created_at", "promised_slot",
#     "cod_amount", "n_items", "address_text", "courier_notes", "delivered_at", "failed",
# ]
FEATURE_COLUMNS: list[str] = [
    "customer_id", "merchant_id", "area", "promised_slot",
    "cod_amount", "n_items", "address_text",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df (same index and row order) with two new columns.

    Needs customer_id, created_at and failed.
    - customer_prior_parcels (int): the same customer's parcels with created_at strictly
      earlier than this row.
    - customer_prior_fail_rate (float): share of those that failed; NaN when there are none.
    Parcels with the same timestamp are not prior to each other. The result must not depend
    on the input row order. Never modify the input.
    """
    result = df.copy()
    cols = ['customer_id', 'created_at']
    grouped = (
        df.groupby(cols, sort=True, dropna=False)['failed']
        .agg(['size', 'sum'])
    )

    prior_count = grouped.groupby(level=0)['size'].cumsum() - grouped['size']
    prior_failures = grouped.groupby(level=0)['sum'].cumsum() - grouped['sum']

    prior_rate = prior_failures / prior_count.where(prior_count > 0)

    keys = pd.MultiIndex.from_frame(df[cols])

    result['customer_prior_parcels'] = (
        prior_count.reindex(keys).fillna(0).to_numpy().astype(int)
    )

    result['customer_prior_fail_rate'] = (
        prior_rate.reindex(keys).to_numpy()
    )

    return result
    # raise NotImplementedError


def time_split(df: pd.DataFrame, time_col: str = "created_at",
               test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test).

    ValueError unless 0 < test_fraction < 1. Sort by time; n_test = ceil(len(df) * test_fraction);
    the boundary is the timestamp of the first of the last n_test rows. Rows at or after the
    boundary are test, rows before it are train.
    """
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be between 0 and 1')

    if len(df) == 0:
        return df.copy(), df.copy()

    n_test = math.ceil(len(df) * test_fraction)
    sorted_df = df.sort_values(time_col)
    boundary = sorted_df.iloc[-n_test][time_col]
    train = sorted_df[sorted_df[time_col] < boundary].copy()
    test = sorted_df[sorted_df[time_col] >= boundary].copy()

    return train, test
    # raise NotImplementedError


def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an UNFITTED Pipeline.

    Input: a DataFrame with at least FEATURE_COLUMNS plus the two history columns; ignore any
    other column. The final step must support predict_proba. Must not fail on unseen
    categories, missing numbers, missing categories or empty address strings. Same seed and
    same data must give identical predictions.
    """
    numeric_features = [
        "cod_amount", "n_items", "customer_prior_parcels", "customer_prior_fail_rate",
    ]

    categorical_features = [
        "customer_id","merchant_id", "area", "promised_slot", "address_text",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__missing__",
                keep_empty_features=True,
            ),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            LogisticRegression(
                C=1,
                class_weight="balanced",
                max_iter=2000,
                random_state=seed,
            ),
        ),
    ])

    return pipeline
    # raise NotImplementedError


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """{"roc_auc", "pr_auc", "brier", "base_rate", "n"} from model.predict_proba(test_df)[:, 1]
    against test_df["failed"] (roc_auc_score, average_precision_score, brier_score_loss)."""
    
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    prob = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
        "brier": brier_score_loss(y_test, prob),
        "base_rate": y_test.mean(),
        "n": len(y_test),
    }
    # raise NotImplementedError


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    """Fit build_pipeline(seed) and DummyClassifier(strategy="prior") on train_df only, evaluate
    both on test_df, return {"model": {...}, "dummy": {...}, "lift_pr_auc": model - dummy PR-AUC}."""
    
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    # Train the main model
    model = build_pipeline(seed)
    model.fit(X_train, y_train)

    model_results = evaluate(model, test_df)

    # Train a baseline that always predicts the training failure rate
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)

    dummy_results = evaluate(dummy, test_df)

    return {
        "model": model_results,
        "dummy": dummy_results,
        "lift_pr_auc": model_results['pr_auc'] - dummy_results['pr_auc'],
    }
    # raise NotImplementedError
