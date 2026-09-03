"""Leakage-safe LASSO feature selection for preprocessed tabular data."""

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted


FEATURE_GROUP_COLORS = {
    "clinical": "#4C78A8",
    "kinetic": "#F58518",
    "morphology": "#54A24B",
    "heterogeneity": "#E45756",
}


def save_lasso_feature_importance_chart(
    report,
    output_path,
    title,
    importance_column="lasso_importance",
    top_n=30,
    frequency_column=None,
):
    """Save a horizontal importance chart from a fold or stability report."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    required_columns = {"feature_name", "feature_group", importance_column}
    if frequency_column is not None:
        required_columns.add(frequency_column)
    missing_columns = required_columns.difference(report.columns)
    if missing_columns:
        raise ValueError(
            f"LASSO chart report is missing columns: {sorted(missing_columns)}"
        )

    plotted = report.copy()
    if "selected" in plotted.columns:
        plotted = plotted.loc[plotted["selected"].astype(bool)]
    if "selection_count" in plotted.columns:
        plotted = plotted.loc[plotted["selection_count"] > 0]
    plotted = plotted.loc[
        np.isfinite(plotted[importance_column].to_numpy(dtype=np.float64))
    ]
    if plotted.empty:
        raise ValueError("LASSO chart report contains no selected finite features.")

    plotted = (
        plotted.sort_values(
            [importance_column, "feature_name"],
            ascending=[False, True],
            kind="stable",
        )
        .head(top_n)
        .sort_values(importance_column, ascending=True, kind="stable")
    )
    display_names = [
        str(name).split("__", 1)[-1]
        for name in plotted["feature_name"]
    ]
    colors = [
        FEATURE_GROUP_COLORS.get(group, "#7F7F7F")
        for group in plotted["feature_group"]
    ]
    figure_height = min(15.0, max(4.5, 0.34 * len(plotted) + 2.0))
    fig, ax = plt.subplots(figsize=(12, figure_height))
    bars = ax.barh(
        np.arange(len(plotted)),
        plotted[importance_column].to_numpy(dtype=np.float64),
        color=colors,
    )
    ax.set_yticks(np.arange(len(plotted)), labels=display_names)
    ax.set_xlabel("Absolute LASSO coefficient")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    if frequency_column is not None:
        frequencies = plotted[frequency_column].to_numpy(dtype=np.float64)
        for bar, frequency in zip(bars, frequencies):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  selected in {frequency:.0%} of folds",
                va="center",
                fontsize=8,
            )

    present_groups = [
        group for group in FEATURE_GROUP_COLORS
        if group in set(plotted["feature_group"])
    ]
    ax.legend(
        handles=[
            Patch(color=FEATURE_GROUP_COLORS[group], label=group)
            for group in present_groups
        ],
        title="Feature family",
        loc="lower right",
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


class LassoFeatureSelector(TransformerMixin, BaseEstimator):
    """Select predictors with non-zero L1-logistic-regression coefficients.

    LASSO is a regression term; for the classification target used by this
    project, its direct analogue is logistic regression with an L1 penalty.
    The regularization strength is selected by an inner cross-validation loop.
    When patient groups are provided, no patient is split across inner folds.
    """

    def __init__(
        self,
        cv_folds=5,
        cs=20,
        max_iter=5000,
        tolerance=1e-4,
        min_features=1,
        coefficient_threshold=1e-8,
        n_jobs=1,
        random_state=42,
    ):
        if cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        if cs < 1:
            raise ValueError("cs must be at least 1.")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1.")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive.")
        if min_features < 1:
            raise ValueError("min_features must be at least 1.")
        if coefficient_threshold < 0:
            raise ValueError("coefficient_threshold cannot be negative.")
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be zero.")

        self.cv_folds = int(cv_folds)
        self.cs = int(cs)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.min_features = int(min_features)
        self.coefficient_threshold = float(coefficient_threshold)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)

    @staticmethod
    def _validate_training_arrays(features, labels, sample_weight, groups):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        if features.ndim != 2 or features.shape[1] == 0:
            raise ValueError("features must have shape (samples, nonzero_features).")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("labels must be one-dimensional and align with features.")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values.")

        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("LASSO feature selection requires at least two classes.")

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (len(features),):
                raise ValueError("sample_weight must align with features.")
            if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight < 0):
                raise ValueError("sample_weight must be finite and non-negative.")
            if not np.any(sample_weight > 0):
                raise ValueError("sample_weight must contain a positive value.")

        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape != (len(features),):
                raise ValueError("groups must align with features.")
            if pd.isna(groups).any():
                raise ValueError("groups cannot contain missing values.")

        return features, labels, sample_weight, groups, classes

    def _make_cv_splits(self, features, labels, groups, classes):
        if groups is None:
            class_counts = np.array([np.sum(labels == label) for label in classes])
            effective_folds = min(self.cv_folds, int(class_counts.min()))
            if effective_folds < 2:
                raise ValueError(
                    "Every class needs at least two training rows for LASSO CV."
                )
            splitter = StratifiedKFold(
                n_splits=effective_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = list(splitter.split(features, labels))
        else:
            group_labels = pd.DataFrame({"group": groups, "label": labels})
            labels_per_group = group_labels.groupby("group", sort=False)["label"].nunique()
            if (labels_per_group > 1).any():
                raise ValueError("Each patient group must have exactly one label.")
            unique_groups = group_labels.drop_duplicates()
            groups_per_class = np.array(
                [
                    unique_groups.loc[
                        unique_groups["label"] == label, "group"
                    ].nunique()
                    for label in classes
                ]
            )
            effective_folds = min(self.cv_folds, int(groups_per_class.min()))
            if effective_folds < 2:
                raise ValueError(
                    "Every class needs at least two patient groups for LASSO CV."
                )
            splitter = StratifiedGroupKFold(
                n_splits=effective_folds,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = list(splitter.split(features, labels, groups))

        self.effective_cv_folds_ = effective_folds
        return splits

    def fit(self, features, labels, sample_weight=None, groups=None):
        """Fit on one outer training fold and return this selector."""
        (
            features,
            labels,
            sample_weight,
            groups,
            classes,
        ) = self._validate_training_arrays(
            features,
            labels,
            sample_weight,
            groups,
        )
        if self.min_features > features.shape[1]:
            raise ValueError(
                "min_features cannot exceed the number of input features."
            )

        cv_splits = self._make_cv_splits(features, labels, groups, classes)
        scoring = "roc_auc" if len(classes) == 2 else "roc_auc_ovr_weighted"
        model_parameters = {
            "Cs": self.cs,
            "cv": cv_splits,
            "solver": "saga",
            "scoring": scoring,
            "tol": self.tolerance,
            "max_iter": self.max_iter,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "refit": True,
        }
        logistic_parameters = inspect.signature(LogisticRegressionCV).parameters
        if (
            "penalty" in logistic_parameters
            and logistic_parameters["penalty"].default != "deprecated"
        ):
            # scikit-learn 1.7, which is pinned by the cluster launchers.
            model_parameters["penalty"] = "l1"
        else:
            # scikit-learn 1.8+ expresses the same L1 model this way.
            model_parameters["l1_ratios"] = (1.0,)
        if "use_legacy_attributes" in logistic_parameters:
            model_parameters["use_legacy_attributes"] = False

        self.model_ = LogisticRegressionCV(
            **model_parameters,
        )
        self.model_.fit(features, labels, sample_weight=sample_weight)

        coefficients = np.asarray(self.model_.coef_, dtype=np.float64)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        importances = np.max(np.abs(coefficients), axis=0)
        support = importances > self.coefficient_threshold

        # A very strong CV penalty can shrink every coefficient to zero. Keep
        # the strongest requested number so downstream classifiers always have
        # a valid input matrix, and make that fallback explicit in the report.
        fallback_count = max(0, self.min_features - int(support.sum()))
        if fallback_count:
            order = np.argsort(-importances, kind="stable")
            support[order[: self.min_features]] = True

        self.n_features_in_ = features.shape[1]
        self.classes_ = np.asarray(self.model_.classes_)
        self.coefficients_ = coefficients
        self.feature_importances_ = importances
        self.support_ = support
        self.selected_indices_ = np.flatnonzero(support)
        self.minimum_feature_fallback_used_ = bool(fallback_count)
        return self

    def transform(self, features):
        check_is_fitted(self, "support_")
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.n_features_in_:
            raise ValueError(
                "features must be two-dimensional with the same columns used "
                "to fit the LASSO selector."
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("features must contain only finite values.")
        return features[:, self.support_]

    def fit_transform(self, features, labels, sample_weight=None, groups=None):
        return self.fit(
            features,
            labels,
            sample_weight=sample_weight,
            groups=groups,
        ).transform(features)

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        return self.selected_indices_.copy() if indices else self.support_.copy()

    def get_feature_names_out(self, input_features):
        check_is_fitted(self, "support_")
        input_features = np.asarray(input_features, dtype=object)
        if input_features.shape != (self.n_features_in_,):
            raise ValueError("input_features must match the fitted feature count.")
        return input_features[self.support_].astype(str).tolist()

    @staticmethod
    def _feature_group(feature_name):
        source_name = str(feature_name).split("__", 1)[-1]
        for group in ("kinetic", "morphology", "heterogeneity"):
            if source_name.startswith(f"{group}_"):
                return group
        return "clinical"

    def selection_report(self, feature_names):
        """Return a ranked table of selected features and class coefficients."""
        check_is_fitted(self, "support_")
        feature_names = np.asarray(feature_names, dtype=object)
        if feature_names.shape != (self.n_features_in_,):
            raise ValueError("feature_names must match the fitted feature count.")

        order = np.argsort(-self.feature_importances_, kind="stable")
        ranks = np.empty(self.n_features_in_, dtype=np.int64)
        ranks[order] = np.arange(1, self.n_features_in_ + 1)
        report = pd.DataFrame(
            {
                "feature_name": feature_names.astype(str),
                "feature_group": [
                    self._feature_group(name) for name in feature_names
                ],
                "selected": self.support_,
                "lasso_importance": self.feature_importances_,
                "lasso_rank": ranks,
                "minimum_feature_fallback_used": (
                    self.minimum_feature_fallback_used_
                ),
            }
        )
        coefficient_classes = (
            [self.classes_[-1]]
            if self.coefficients_.shape[0] == 1
            else self.classes_
        )
        for row_index, class_label in enumerate(coefficient_classes):
            report[f"coefficient_class_{class_label}"] = self.coefficients_[row_index]
        return report.sort_values(
            ["selected", "lasso_importance", "feature_name"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)

    @property
    def output_dimension(self):
        check_is_fitted(self, "support_")
        return int(self.support_.sum())
