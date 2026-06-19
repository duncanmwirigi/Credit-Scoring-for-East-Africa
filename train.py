#!/usr/bin/env python3
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.synthetic import generate_synthetic_portfolio
from src.ml.trainer import CreditModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("credit_scoring.train")


def main() -> None:
    config = load_config()
    logger.info("Starting training for %s v%s", config.project_name, config.version)

    dataset = generate_synthetic_portfolio(
        n_samples=config.data["n_samples"],
        default_rate=config.data["default_rate"],
        channel_distribution=config.data["channel_distribution"],
        seed=config.random_state,
    )

    trainer = CreditModelTrainer(config)
    pipeline, metrics = trainer.train(dataset)
    artifacts = trainer.persist(pipeline, dataset, metrics)

    logger.info("Training complete.")
    logger.info("ROC-AUC: %.3f | Gini: %.3f | KS: %.3f", metrics["roc_auc"], metrics["gini"], metrics["ks"])
    logger.info("Model: %s", artifacts.model_path)
    logger.info("Metrics: %s", artifacts.metrics_path)
    logger.info("ROC plot: %s", artifacts.roc_plot_path)


if __name__ == "__main__":
    main()
