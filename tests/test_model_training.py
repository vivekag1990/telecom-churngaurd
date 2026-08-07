"""ML BEHAVIOURAL TESTS -- model TRAINING."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from churnguard.features.engineering import ChurnFeatureEngineer, build_preprocessor


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

        # Handle string y_small
        import pandas as pd

        if pd.api.types.is_string_dtype(y_small):
            y_small = (y_small == "Yes").astype(int)

        pipeline.fit(x_small, y_small)
        train_accuracy = pipeline.score(x_small, y_small)
        assert (
            train_accuracy >= 0.98
        ), f"Cannot memorise 60 rows (acc={train_accuracy:.3f})"
