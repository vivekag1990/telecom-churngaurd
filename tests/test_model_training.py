"""ML BEHAVIOURAL TESTS - model training."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from churnguard.config import SETTINGS
from churnguard.exceptions import ModelTrainingError
from churnguard.features.engineering import ChurnFeatureEngineer, build_preprocessor
from churnguard.models.trainer import ChurnModelTrainer


class TestTrainingContract:
    def test_pipeline_has_the_expected_three_stages(self):
        steps = [name for name, _ in ChurnModelTrainer().build_pipeline().steps]
        assert steps == ["features", "preprocess", "classifier"]

    def test_mismatched_x_and_y_lengths_are_rejected(self, split):
        with pytest.raises(ModelTrainingError, match="refusing to train"):
            ChurnModelTrainer().train(split.x_train, split.y_train.iloc[:-5])

    def test_single_class_target_is_rejected(self, split):
        target = pd.Series(np.zeros(len(split.x_train)))
        with pytest.raises(ModelTrainingError, match="two target classes"):
            ChurnModelTrainer().train(split.x_train, target)

    def test_evaluate_before_train_is_rejected(self, split):
        with pytest.raises(ModelTrainingError, match="before train"):
            ChurnModelTrainer().evaluate(split.x_test, split.y_test)


class TestLearningActuallyHappens:
    def test_model_overfits_a_small_batch(self, split):
        x_small = split.x_train.head(60)
        y_small = split.y_train.head(60)
        pipeline = Pipeline(
            [
                ("features", ChurnFeatureEngineer()),
                ("preprocess", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        min_samples_leaf=1,
                        random_state=0,
                    ),
                ),
            ]
        )
        pipeline.fit(x_small, y_small)
        train_accuracy = pipeline.score(x_small, y_small)
        assert train_accuracy >= 0.98, f"Cannot memorise 60 rows (acc={train_accuracy:.3f})"

    def test_training_loss_decreases_with_model_capacity(self, split):
        losses = []
        for n_estimators in (1, 5, 25, 100):
            pipeline = Pipeline(
                [
                    ("features", ChurnFeatureEngineer()),
                    ("preprocess", build_preprocessor()),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=None,
                            min_samples_leaf=1,
                            random_state=0,
                        ),
                    ),
                ]
            )
            pipeline.fit(split.x_train, split.y_train)
            probabilities = pipeline.predict_proba(split.x_train)[:, 1]
            losses.append(log_loss(split.y_train, probabilities, labels=[0, 1]))
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"
        assert losses == sorted(losses, reverse=True), f"Loss not monotonically falling: {losses}"

    def test_model_beats_the_majority_class_baseline(self, trained_pipeline, split):
        baseline = DummyClassifier(strategy="most_frequent").fit(split.x_train, split.y_train)
        baseline_auc = roc_auc_score(split.y_test, baseline.predict_proba(split.x_test)[:, 1])
        model_auc = roc_auc_score(split.y_test, trained_pipeline.predict_proba(split.x_test)[:, 1])
        assert model_auc > baseline_auc + 0.15

    def test_model_learns_signal_not_noise(self, split, trained_pipeline):
        shuffled_aucs = []
        for seed in (11, 12, 13):
            rng = np.random.default_rng(seed)
            shuffled = pd.Series(
                rng.permutation(split.y_train.to_numpy()),
                index=split.y_train.index,
            )
            trainer = ChurnModelTrainer()
            trainer.train(split.x_train, shuffled)
            shuffled_aucs.append(
                roc_auc_score(
                    split.y_test,
                    trainer.pipeline.predict_proba(split.x_test)[:, 1],
                )
            )
        mean_auc = float(np.mean(shuffled_aucs))
        real_auc = roc_auc_score(split.y_test, trained_pipeline.predict_proba(split.x_test)[:, 1])
        assert abs(mean_auc - 0.5) < 0.12
        assert real_auc - mean_auc > 0.20


class TestReproducibilityAndGates:
    def test_same_seed_gives_identical_predictions(self, split):
        first = ChurnModelTrainer()
        first.train(split.x_train, split.y_train)
        second = ChurnModelTrainer()
        second.train(split.x_train, split.y_train)
        np.testing.assert_allclose(
            first.pipeline.predict_proba(split.x_test)[:, 1],
            second.pipeline.predict_proba(split.x_test)[:, 1],
            rtol=1e-9,
        )

    def test_trained_model_clears_release_gates(self, trained_trainer, split):
        metrics = trained_trainer.evaluate(split.x_test, split.y_test)
        passed, failures = metrics.passes_gates()
        assert passed, f"Quality gates failed: {failures}"

    def test_artifact_round_trips_with_metadata(self, artifact_path):
        payload = joblib.load(artifact_path)
        assert "pipeline" in payload
        assert payload["metadata"]["input_features"] == SETTINGS.all_features
        assert payload["metadata"]["hyperparameters"]["calibration"] == "isotonic"
