"""Leakage-safe preprocessing for tabular features used by fusion models."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ClinicalFeaturePreprocessor:
    """Fit numeric and categorical preprocessing on one training fold only."""

    def __init__(self, continuous_columns, categorical_columns):
        self.continuous_columns = tuple(continuous_columns)
        self.categorical_columns = tuple(categorical_columns)
        overlap = set(self.continuous_columns).intersection(self.categorical_columns)
        if overlap:
            raise ValueError(f"Columns cannot be both continuous and categorical: {sorted(overlap)}")
        if not self.continuous_columns and not self.categorical_columns:
            raise ValueError("At least one tabular feature column is required.")
        self.transformer = None

    @property
    def columns(self):
        return (*self.continuous_columns, *self.categorical_columns)

    def _validate_columns(self, data):
        missing = set(self.columns).difference(data.columns)
        if missing:
            raise ValueError(f"Missing tabular feature columns: {sorted(missing)}")

    def fit(self, train_data):
        self._validate_columns(train_data)
        transformers = []
        if self.continuous_columns:
            continuous_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median", keep_empty_features=True),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(
                ("continuous", continuous_pipeline, list(self.continuous_columns))
            )

        if self.categorical_columns:
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    ),
                    (
                        "one_hot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(
                ("categorical", categorical_pipeline, list(self.categorical_columns))
            )

        self.transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self.transformer.fit(train_data)
        return self

    def transform(self, data):
        if self.transformer is None:
            raise RuntimeError("Fit ClinicalFeaturePreprocessor before transform().")
        self._validate_columns(data)
        transformed = self.transformer.transform(data)
        transformed = np.asarray(transformed, dtype=np.float32)
        if not np.all(np.isfinite(transformed)):
            raise ValueError("Tabular preprocessing produced non-finite values.")
        return transformed

    def fit_transform(self, train_data):
        return self.fit(train_data).transform(train_data)

    def get_feature_names_out(self):
        if self.transformer is None:
            raise RuntimeError("Fit ClinicalFeaturePreprocessor before requesting names.")
        return self.transformer.get_feature_names_out().tolist()

    def add_feature_vectors(
        self,
        train_data,
        validation_data,
        output_column="clinical_features",
    ):
        """Return copies with dense float32 vectors created from the train fit."""
        train_features = self.fit_transform(train_data)
        validation_features = self.transform(validation_data)

        train_output = train_data.copy()
        validation_output = validation_data.copy()
        train_output[output_column] = [row.copy() for row in train_features]
        validation_output[output_column] = [row.copy() for row in validation_features]
        return train_output, validation_output

    @property
    def output_dimension(self):
        return len(self.get_feature_names_out())
