from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import AppConfig
from src.features.engineering import FEATURE_COLUMNS, build_feature_matrix
from src.ml.metrics import classification_report

logger = logging.getLogger(__name__)


@dataclass
class TrainingArtifacts:
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    feature_importance_path: Path
    roc_plot_path: Path


class CreditModelTrainer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

    def _build_estimator(self):
        model_type = self.config.training["model_type"]
        if model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=500,
                class_weight=self.config.training["class_weight"],
                random_state=self.config.random_state,
            )
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=self.config.random_state,
        )

    def _build_pipeline(self) -> Pipeline:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        preprocessor = ColumnTransformer(
            transformers=[("numeric", Pipeline(numeric_steps), FEATURE_COLUMNS)],
            remainder="drop",
        )
        estimator = self._build_estimator()
        if self.config.training.get("calibration", True):
            estimator = CalibratedClassifierCV(estimator, cv=3, method="sigmoid")
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

    def train(self, dataset: pd.DataFrame) -> tuple[Pipeline, dict]:
        features = build_feature_matrix(dataset)
        target = dataset["default_12m"].to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.config.training["test_size"],
            random_state=self.config.random_state,
            stratify=target,
        )

        pipeline = self._build_pipeline()
        logger.info("Training %s model on %s samples", self.config.training["model_type"], len(x_train))
        pipeline.fit(x_train, y_train)

        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics = classification_report(y_test, probabilities)
        metrics["train_size"] = len(x_train)
        metrics["test_size"] = len(x_test)
        metrics["default_rate"] = float(target.mean())
        metrics["model_type"] = self.config.training["model_type"]
        metrics["trained_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Validation metrics — AUC: %.3f, Gini: %.3f, KS: %.3f",
            metrics["roc_auc"],
            metrics["gini"],
            metrics["ks"],
        )
        return pipeline, metrics

    def _feature_importance(self, pipeline: Pipeline) -> pd.DataFrame:
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "calibrated_classifiers_"):
            base = classifier.calibrated_classifiers_[0].estimator
        else:
            base = classifier

        if hasattr(base, "feature_importances_"):
            values = base.feature_importances_
        elif hasattr(base, "coef_"):
            values = np.abs(base.coef_).ravel()
        else:
            return pd.DataFrame(columns=["feature", "importance"])

        return (
            pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def _save_roc_curve(self, pipeline: Pipeline, dataset: pd.DataFrame) -> Path:
        from sklearn.metrics import RocCurveDisplay

        features = build_feature_matrix(dataset)
        target = dataset["default_12m"].to_numpy()
        _, x_test, _, y_test = train_test_split(
            features,
            target,
            test_size=self.config.training["test_size"],
            random_state=self.config.random_state,
            stratify=target,
        )
        probabilities = pipeline.predict_proba(x_test)[:, 1]

        fig, ax = plt.subplots(figsize=(7, 5))
        RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
        ax.set_title("Credit Model ROC Curve — M-Pesa / SACCO / Bank Portfolio")
        output = self.config.reports_dir / "roc_curve.png"
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output

    def persist(self, pipeline: Pipeline, dataset: pd.DataFrame, metrics: dict) -> TrainingArtifacts:
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_path = self.config.model_dir / f"credit_model_{version}.joblib"
        metadata_path = self.config.model_dir / f"credit_model_{version}.json"
        metrics_path = self.config.reports_dir / "training_metrics.json"
        feature_importance_path = self.config.reports_dir / "feature_importance.csv"
        roc_plot_path = self._save_roc_curve(pipeline, dataset)

        joblib.dump(pipeline, model_path)
        importance = self._feature_importance(pipeline)
        importance.to_csv(feature_importance_path, index=False)

        metadata = {
            "project": self.config.project_name,
            "version": self.config.version,
            "model_file": model_path.name,
            "feature_columns": FEATURE_COLUMNS,
            "scorecard": asdict(self.config.scorecard),
            "decision_bands": self.config.decision_bands,
            "metrics": metrics,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (self.config.model_dir / "latest_model.txt").write_text(model_path.name, encoding="utf-8")

        logger.info("Saved model to %s", model_path)
        return TrainingArtifacts(
            model_path=model_path,
            metadata_path=metadata_path,
            metrics_path=metrics_path,
            feature_importance_path=feature_importance_path,
            roc_plot_path=roc_plot_path,
        )
