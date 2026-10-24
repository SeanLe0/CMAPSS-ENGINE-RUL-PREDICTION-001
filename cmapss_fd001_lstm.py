"""
CMAPSS FD001 Remaining Useful Life (RUL) prediction with PyTorch LSTM.

What this script does
---------------------
1. Loads NASA CMAPSS FD001 train/test/RUL files.
2. Splits TRAINING ENGINE IDs into 80% train and 20% validation.
3. Creates capped training targets: RUL = min(final_cycle - cycle, 125).
4. Simulates truncated validation trajectories so validation evaluates RUL
   at a pre-failure cutoff, rather than at failure (where RUL would be zero).
5. Fits feature scaling on training engines only.
6. Trains separate LSTMs with hidden sizes 2, 4, 8, 16, 32, 64, and 128.
7. Selects the hidden size with the lowest validation final-cycle RMSE.
8. Evaluates that selected model on the official test set and calculates
   RMSE plus the standard asymmetric NASA CMAPSS score.

Install dependencies in your project environment:
    python -m pip install numpy pandas scikit-learn torch

Expected files:
    D:\\PROJECT\\20260815 CMAPSS JET ENGINE RUL\\CMAPSSData\\train_FD001.txt
    D:\\PROJECT\\20260815 CMAPSS JET ENGINE RUL\\CMAPSSData\\test_FD001.txt
    D:\\PROJECT\\20260815 CMAPSS JET ENGINE RUL\\CMAPSSData\\RUL_FD001.txt

Notes
-----
- The raw CMAPSS files have 26 columns:
  engine ID, cycle, 3 operating settings, and 21 sensor measurements.
- Engine ID and cycle are not model input features. The remaining 24 columns
  are used as input features.
- Training uses a capped target (maximum 125). Official test RUL labels are
  used as supplied, without capping, for final evaluation.
- This is a learning-oriented baseline, not a claim of state-of-the-art
  performance. Results depend on the random split and validation cutoffs.
"""

from pathlib import Path
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


# ----------------------------- Configuration -----------------------------

DATA_DIR = Path(
    r"D:\PROJECT\20260815 CMAPSS JET ENGINE RUL\CMAPSSData"
)

TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.80
RUL_CAP = 125
HIDDEN_SIZES = [2, 4, 8, 16, 32, 64, 128]
LEARNING_RATE = 0.001
EPOCHS = 30

# CPU is sufficient for this small baseline.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------ Reproducibility ---------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------- Loading --------------------------------

COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
FEATURE_COLUMNS = COLUMN_NAMES[2:]  # 3 settings + 21 sensors = 24 features


def load_cmapss_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}\n"
            "Check DATA_DIR at the top of this script."
        )

    frame = pd.read_csv(path, sep=r"\s+", header=None)
    # Some CMAPSS copies contain trailing whitespace columns. Remove
    # columns that are entirely empty, if any.
    frame = frame.dropna(axis=1, how="all")

    if frame.shape[1] != 26:
        raise ValueError(
            f"{path.name} should have 26 columns after removing empty "
            f"columns, but has {frame.shape[1]}."
        )

    frame.columns = COLUMN_NAMES
    frame = frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    return frame


def add_training_rul(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Add per-cycle capped RUL labels to complete training trajectories."""
    result = frame.copy()
    final_cycle = result.groupby("engine_id")["cycle"].transform("max")
    result["rul"] = np.minimum(final_cycle - result["cycle"], cap)
    return result


def split_engine_ids(
    frame: pd.DataFrame,
    train_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split whole engine IDs, never individual rows."""
    engine_ids = np.array(sorted(frame["engine_id"].unique()))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(engine_ids)

    split_at = int(round(len(shuffled) * train_fraction))
    split_at = min(max(split_at, 1), len(shuffled) - 1)

    train_ids = sorted(shuffled[:split_at].tolist())
    validation_ids = sorted(shuffled[split_at:].tolist())
    return train_ids, validation_ids


# ------------------------------- Preparation -----------------------------

def fit_scaler(train_frame: pd.DataFrame) -> StandardScaler:
    """Fit scaling statistics on training-engine rows only."""
    scaler = StandardScaler()
    scaler.fit(train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    return scaler


def scaled_sequence(
    engine_frame: pd.DataFrame,
    scaler: StandardScaler,
) -> np.ndarray:
    return scaler.transform(
        engine_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    ).astype(np.float32)


def make_validation_sequences(
    full_training_frame: pd.DataFrame,
    validation_ids: list[int],
    scaler: StandardScaler,
    seed: int,
) -> list[tuple[np.ndarray, float, int]]:
    """
    Simulate one truncated, pre-failure observation for each held-out engine.

    Each tuple is (scaled sequence up to cutoff, uncapped true RUL at cutoff,
    engine ID). A random cutoff is used so final-cycle validation does not
    trivially compare predictions with RUL=0 at the failure cycle.
    """
    rng = np.random.default_rng(seed)
    examples = []

    for engine_id in validation_ids:
        engine = (
            full_training_frame[
                full_training_frame["engine_id"] == engine_id
            ]
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        # Need at least two observations to create a pre-failure cutoff.
        if len(engine) < 2:
            continue

        # cutoff_index is inclusive. It must be before the final row.
        cutoff_index = int(rng.integers(0, len(engine) - 1))
        observed = engine.iloc[: cutoff_index + 1]
        final_cycle = int(engine["cycle"].max())
        cutoff_cycle = int(observed["cycle"].iloc[-1])
        true_rul_at_cutoff = float(final_cycle - cutoff_cycle)

        examples.append(
            (
                scaled_sequence(observed, scaler),
                true_rul_at_cutoff,
                int(engine_id),
            )
        )

    if not examples:
        raise RuntimeError("No validation sequences were created.")

    return examples


def make_official_test_sequences(
    test_frame: pd.DataFrame,
    official_rul: np.ndarray,
    scaler: StandardScaler,
) -> list[tuple[np.ndarray, float, int]]:
    """
    Make one sequence per test engine and attach the supplied RUL label.
    RUL file row order corresponds to test engine ID order (1, 2, ...).
    """
    engine_ids = sorted(test_frame["engine_id"].unique())

    if len(engine_ids) != len(official_rul):
        raise ValueError(
            f"Found {len(engine_ids)} test engines but "
            f"{len(official_rul)} RUL labels."
        )

    examples = []
    for index, engine_id in enumerate(engine_ids):
        engine = (
            test_frame[test_frame["engine_id"] == engine_id]
            .sort_values("cycle")
            .reset_index(drop=True)
        )
        examples.append(
            (
                scaled_sequence(engine, scaler),
                float(official_rul[index]),
                int(engine_id),
            )
        )
    return examples


# --------------------------------- Model ----------------------------------

class RULLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        sequence shape: [1, sequence_length, input_size]
        returns one prediction per time step: [sequence_length]
        """
        output, _ = self.lstm(sequence)
        predictions = self.output_layer(output).squeeze(-1)
        return predictions.squeeze(0)


# -------------------------------- Training --------------------------------

def train_one_model(
    train_frame: pd.DataFrame,
    train_ids: list[int],
    scaler: StandardScaler,
    hidden_size: int,
    seed: int,
) -> RULLSTM:
    set_seed(seed + hidden_size)

    model = RULLSTM(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=hidden_size,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    engines_by_id = {
        int(engine_id): engine.sort_values("cycle").reset_index(drop=True)
        for engine_id, engine in train_frame[
            train_frame["engine_id"].isin(train_ids)
        ].groupby("engine_id")
    }

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        engine_order = train_ids.copy()
        random.Random(seed + hidden_size * 1000 + epoch).shuffle(engine_order)

        for engine_id in engine_order:
            engine = engines_by_id[engine_id]
            x_array = scaled_sequence(engine, scaler)
            y_array = engine["rul"].to_numpy(dtype=np.float32)

            x_tensor = torch.from_numpy(x_array).unsqueeze(0).to(DEVICE)
            y_tensor = torch.from_numpy(y_array).to(DEVICE)

            optimizer.zero_grad()
            predictions = model(x_tensor)

            # Calculate loss at every cycle in this engine trajectory.
            loss = loss_function(predictions, y_tensor)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(epoch_losses))
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print(
                f"  hidden={hidden_size:2d} | "
                f"epoch {epoch + 1:02d}/{EPOCHS} | "
                f"mean engine loss={mean_loss:.4f}"
            )

    return model


# ------------------------------- Evaluation -------------------------------

@torch.no_grad()
def predict_final_rul(
    model: RULLSTM,
    examples: list[tuple[np.ndarray, float, int]],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    model.eval()
    predictions = []
    targets = []
    engine_ids = []

    for sequence, target, engine_id in examples:
        x_tensor = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)
        per_cycle_predictions = model(x_tensor)
        final_prediction = float(per_cycle_predictions[-1].cpu().item())

        predictions.append(final_prediction)
        targets.append(float(target))
        engine_ids.append(engine_id)

    return (
        np.asarray(predictions, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        engine_ids,
    )


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def nasa_cmapss_score(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Standard asymmetric CMAPSS score.
    d = prediction - true RUL.
    Positive d (overestimation) uses denominator 10;
    negative d (underestimation) uses denominator 13.
    """
    errors = predictions - targets
    scores = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )
    return float(np.sum(scores))


# ---------------------------------- Main ----------------------------------

def main() -> None:
    set_seed(RANDOM_SEED)
    print(f"Device: {DEVICE}")
    print("Loading CMAPSS FD001 files...")

    train_raw = load_cmapss_file(TRAIN_FILE)
    test_raw = load_cmapss_file(TEST_FILE)
    train_with_rul = add_training_rul(train_raw, RUL_CAP)

    # The official RUL file contains one value per test engine.
    official_rul_frame = pd.read_csv(RUL_FILE, sep=r"\s+", header=None)
    official_rul = official_rul_frame.iloc[:, 0].to_numpy(dtype=np.float64)

    train_ids, validation_ids = split_engine_ids(
        train_with_rul,
        train_fraction=TRAIN_FRACTION,
        seed=RANDOM_SEED,
    )
    print(
        f"Training engines: {len(train_ids)} | "
        f"validation engines: {len(validation_ids)}"
    )

    # Fit preprocessing only on training-engine data.
    train_rows = train_with_rul[
        train_with_rul["engine_id"].isin(train_ids)
    ].copy()
    scaler = fit_scaler(train_rows)

    validation_examples = make_validation_sequences(
        full_training_frame=train_with_rul,
        validation_ids=validation_ids,
        scaler=scaler,
        seed=RANDOM_SEED + 1,
    )

    best_model = None
    best_hidden_size = None
    best_validation_rmse = float("inf")
    results = []

    for hidden_size in HIDDEN_SIZES:
        print(f"\nTraining candidate with {hidden_size} hidden units...")
        model = train_one_model(
            train_frame=train_with_rul,
            train_ids=train_ids,
            scaler=scaler,
            hidden_size=hidden_size,
            seed=RANDOM_SEED,
        )

        validation_predictions, validation_targets, _ = predict_final_rul(
            model, validation_examples
        )
        validation_rmse = rmse(validation_predictions, validation_targets)
        results.append((hidden_size, validation_rmse))

        print(
            f"Validation final-cycle RMSE, hidden={hidden_size}: "
            f"{validation_rmse:.4f}"
        )

        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
            best_hidden_size = hidden_size
            best_model = copy.deepcopy(model).cpu()

    assert best_model is not None
    print("\nValidation results:")
    for hidden_size, score in results:
        print(f"  hidden={hidden_size:2d}: RMSE={score:.4f}")

    print(
        f"\nSelected hidden size: {best_hidden_size} "
        f"(validation RMSE={best_validation_rmse:.4f})"
    )

    # Evaluate the selected model once on the official test set.
    best_model = best_model.to(DEVICE)
    test_examples = make_official_test_sequences(
        test_frame=test_raw,
        official_rul=official_rul,
        scaler=scaler,
    )
    test_predictions, test_targets, test_engine_ids = predict_final_rul(
        best_model, test_examples
    )

    test_rmse = rmse(test_predictions, test_targets)
    test_score = nasa_cmapss_score(test_predictions, test_targets)

    print("\nOfficial test results:")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  NASA CMAPSS score: {test_score:.4f}")

    # Plot final-cycle predictions against the official RUL labels.
    # Apply the same min-max transform to both arrays so y=x remains the
    # perfect-prediction reference, including if a prediction is negative.
    shared_min = min(float(np.min(test_targets)), float(np.min(test_predictions)))
    shared_max = max(float(np.max(test_targets)), float(np.max(test_predictions)))
    normalization_range = max(shared_max - shared_min, 1e-12)
    actual_normalized = (test_targets - shared_min) / normalization_range
    predicted_normalized = (test_predictions - shared_min) / normalization_range

    # Fit a straight trend line through the predicted-vs-actual points.
    slope, intercept = np.polyfit(actual_normalized, predicted_normalized, 1)
    x_line = np.linspace(0.0, 1.0, 200)
    prediction_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        actual_normalized,
        predicted_normalized,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.4,
        label="Test engines (final observed cycle)",
    )
    ax.plot(
        x_line,
        x_line,
        linestyle="--",
        linewidth=2,
        label="Perfect prediction (y = x)",
    )
    ax.plot(
        x_line,
        prediction_line,
        linewidth=2,
        label=f"Prediction trend (y = {slope:.2f}x + {intercept:.2f})",
    )
    ax.set(
        xlabel="Actual RUL (normalized)",
        ylabel="Predicted RUL (normalized)",
        title="CMAPSS FD001: Predicted vs Actual RUL",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = Path("cmapss_fd001_predicted_vs_actual.png")
    fig.savefig(plot_path, dpi=200)
    print(f"Prediction plot saved to: {plot_path.resolve()}")
    plt.show()
    plt.close(fig)

    results_frame = pd.DataFrame(
        {
            "engine_id": test_engine_ids,
            "predicted_rul": test_predictions,
            "true_rul": test_targets,
            "error_pred_minus_true": test_predictions - test_targets,
        }
    )
    output_path = Path("cmapss_fd001_test_predictions.csv")
    results_frame.to_csv(output_path, index=False)
    print(f"Per-engine predictions saved to: {output_path.resolve()}")

    # Save the selected model's parameters.
    model_path = Path("cmapss_fd001_lstm_best.pt")
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "input_size": len(FEATURE_COLUMNS),
            "hidden_size": best_hidden_size,
            "rul_cap": RUL_CAP,
            "feature_columns": FEATURE_COLUMNS,
            "random_seed": RANDOM_SEED,
        },
        model_path,
    )
    print(f"Selected model saved to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
