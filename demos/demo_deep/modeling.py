from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agentic_ml_loop.candidate_result import CandidateRunner, FittedCandidate
from agentic_ml_loop.splits import SearchSplits
from demos.common import (
    RANDOM_STATE,
    DemoCandidate,
    DemoScoring,
    classification_metrics,
    fitted_candidate,
)
from demos.demo_deep.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    feature_columns,
)

SCORING = DemoScoring(TARGET_COLUMN, "val_auc", classification_metrics)
MAX_EPOCHS = 120
PATIENCE = 12
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_COLUMNS),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(CATEGORICAL_COLUMNS),
            ),
        ]
    )


class TabularMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for index, hidden in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0 and index < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class MLPTrainingData:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray


@dataclass(frozen=True)
class MLPArchitecture:
    hidden_dims: Sequence[int]
    dropout: float


def _predict_proba(model: TabularMLP, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32))
        logits = model(tensor)
        return torch.sigmoid(logits).numpy()


def _train_mlp(
    training_data: MLPTrainingData, architecture: MLPArchitecture
) -> tuple[TabularMLP, int]:
    torch.manual_seed(RANDOM_STATE)
    model = TabularMLP(
        training_data.train_x.shape[1],
        architecture.hidden_dims,
        architecture.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    x_tensor = torch.from_numpy(training_data.train_x.astype(np.float32))
    y_tensor = torch.from_numpy(training_data.train_y.astype(np.float32))
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    best_val_auc = float("-inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        val_probs = _predict_proba(model, training_data.val_x)
        if len(np.unique(training_data.val_y)) < 2:
            val_auc = float("-inf")
        else:
            val_auc = float(roc_auc_score(training_data.val_y, val_probs))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def _run_logreg_baseline(splits: SearchSplits) -> FittedCandidate:
    cols = feature_columns()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _preprocessor()),
            (
                "estimator",
                LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver="lbfgs"),
            ),
        ]
    )
    pipeline.fit(splits.train[cols], splits.train[TARGET_COLUMN])
    return fitted_candidate(
        splits,
        SCORING,
        DemoCandidate(
            candidate_id="logreg-baseline",
            model_family="logistic_regression",
            feature_set="base",
            notes="Linear baseline on scaled numeric features and one-hot channel.",
            hyperparameters={"max_iter": 500, "random_state": RANDOM_STATE, "solver": "lbfgs"},
            selected_features=cols,
        ),
        lambda rows: pipeline.predict_proba(rows[cols])[:, 1],
    )


def _run_mlp(
    splits: SearchSplits,
    *,
    candidate_id: str,
    hidden_dims: Sequence[int],
    dropout: float,
    notes: str,
) -> FittedCandidate:
    cols = feature_columns()
    preprocessor = _preprocessor()
    model, best_epoch = _train_mlp(
        MLPTrainingData(
            train_x=np.asarray(preprocessor.fit_transform(splits.train[cols])),
            train_y=splits.train[TARGET_COLUMN].to_numpy(dtype=np.float32),
            val_x=np.asarray(preprocessor.transform(splits.validation[cols])),
            val_y=splits.validation[TARGET_COLUMN].to_numpy(dtype=np.float32),
        ),
        MLPArchitecture(hidden_dims=hidden_dims, dropout=dropout),
    )
    return fitted_candidate(
        splits,
        SCORING,
        DemoCandidate(
            candidate_id=candidate_id,
            model_family="pytorch_mlp",
            feature_set="base",
            notes=notes,
            hyperparameters={
                "hidden_dims": list(hidden_dims),
                "dropout": dropout,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "best_epoch": best_epoch,
                "random_state": RANDOM_STATE,
            },
            selected_features=cols,
        ),
        lambda rows: _predict_proba(model, np.asarray(preprocessor.transform(rows[cols]))),
    )


def run_mlp_shallow(splits: SearchSplits) -> FittedCandidate:
    return _run_mlp(
        splits,
        candidate_id="mlp-shallow",
        hidden_dims=(32,),
        dropout=0.0,
        notes="Single hidden layer (32 units). Early stopping on validation AUC.",
    )


def run_mlp_deep(splits: SearchSplits) -> FittedCandidate:
    return _run_mlp(
        splits,
        candidate_id="mlp-deep",
        hidden_dims=(64, 32),
        dropout=0.2,
        notes="Two hidden layers with dropout. Targets the nonlinear ring + wave boundary.",
    )


def run_mlp_wide(splits: SearchSplits) -> FittedCandidate:
    return _run_mlp(
        splits,
        candidate_id="mlp-wide",
        hidden_dims=(128, 64),
        dropout=0.1,
        notes="Wider MLP for extra capacity on the interaction terms.",
    )


CANDIDATE_RUNNERS: dict[str, CandidateRunner] = {
    "logreg-baseline": _run_logreg_baseline,
    "mlp-shallow": run_mlp_shallow,
    "mlp-deep": run_mlp_deep,
    "mlp-wide": run_mlp_wide,
}
