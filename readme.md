# Tool-wear prediction

Code for predicting the cumulative flank wear of CNC milling tools from
high-frequency sensor data (accelerometers, acoustic emission) and controller
data (loads, feed rate, spindle speed), developed on the
[PHM Asia Pacific 2025 Data Challenge](https://phmap25.org/datachallenge.html).

A causal temporal convolutional network, trained with self-supervision on the
unlabelled sensor windows, encodes each cut; an autoregressive SVR combines
this embedding with controller features to predict wear at the measured cuts.

## Method

1. **Feature extraction** (`scripts/`): each cut is split into *K* time windows
   and summarised with temporal, spectral, wavelet and envelope features.
   Controller signals are summarised over a cumulative window per checkpoint.
2. **Self-supervised encoder** (`toolwear/encoder.py`): a causal TCN is trained
   on the window sequences with masked reconstruction and a temporal triplet
   loss. Being causal, the embedding at a cut only uses information up to it.
3. **Regressor** (`toolwear/regressor.py`): an RBF SVR predicts the wear
   increment between measured cuts from the frozen embedding, controller
   features, anchor features and its own previous prediction
   (autoregressive).
4. **Hyperparameter search** (`toolwear/tuning.py`): a nested Optuna search.
   The outer search picks *K*, the encoder loss weight and the regressor
   hyperparameters using a robust, cut-weighted leave-one-dataset-out CV.
   For each new *(K, λ)* pair, an inner search tunes the TCN.
5. **Automatic termination** (`toolwear/termination.py`): the outer search
   stops early following Makarova et al. (AutoML 2022), using Optuna's
   `RegretBoundEvaluator`. The search stops once the high-probability bound on
   the remaining regret falls below the statistical error of the best trial's
   robust CV score (Nadeau-Bengio variance of the mean plus the variance of the
   fold standard deviation). Beyond that point, further gains in CV score are
   within the estimation error and are unlikely to improve generalisation.

## Repository layout

```
scripts/
  extract_controller_features.py   # controller features per checkpoint
  extract_sensor_features.py       # windowed sensor features for one or more K
  data_loader_trainset.py          # official script from https://phmap25.org/datachallenge.html#data-sharing
  data_loader_evalset.py           # official script from https://phmap25.org/datachallenge.html#data-sharing
toolwear/
  config.py      # configuration dataclasses (serialisable to JSON)
  data.py        # loading and shaping of features and labels
  encoder.py     # causal TCN, SSL losses and training, TCNEncoder
  regressor.py   # SVR pipeline, autoregressive prediction, LODO CV
  model.py       # ToolWearModel: fit / predict / save / load
  tuning.py      # nested Optuna search (Tuner)
  termination.py # automatic termination criterion for the outer search
  reporting.py   # metrics, plots and search diagnostics
  cli.py         # command-line interface
Training_Dataset/     # from https://phmap25.org/datachallenge.html#data-sharing
Evaluation_Dataset/   # from https://phmap25.org/datachallenge.html#data-sharing
```


## Data

The training and evaluation sets of the PHM Asia Pacific 2025 Data Challenge,
and the official data loaders, are available at
https://phmap25.org/datachallenge.html#data-sharing. They are not included in
this repository.

1. Download both datasets and place them in the repository root:

```
   Training_Dataset/
     Controller_Data/
     Sensor_Data/
     trainset_toolwear_measurement.csv
   Evaluation_Dataset/
     Controller_Data/
     Sensor_Data/
     evalset_toolwear_measurement.csv
```

2. Copy the official loaders `data_loader_trainset.py` and
   `data_loader_evalset.py` into `scripts/`, where the feature scripts import
   them from.


## Installation

```bash
pip install -r requirements.txt
```


## Usage

**1. Extract features**

```bash
python scripts/extract_controller_features.py \
    --train-controller-dir Training_Dataset/Controller_Data \
    --eval-controller-dir Evaluation_Dataset/Controller_Data \
    --output-dir features/

python scripts/extract_sensor_features.py \
    --train-controller-dir Training_Dataset/Controller_Data \
    --train-sensor-dir Training_Dataset/Sensor_Data \
    --eval-controller-dir Evaluation_Dataset/Controller_Data \
    --eval-sensor-dir Evaluation_Dataset/Sensor_Data \
    --output-dir features/ -k 1 5 10 20
```

**2. Search hyperparameters** (resumable; writes `best_config.json` and a fitted model)

```bash
python -m toolwear tune \
    --sensor-template "features/features_sensor_train_K{k}.csv" \
    --controller-features features/features_controller_train.csv \
    --labels Training_Dataset/trainset_toolwear_measurement.csv \
    --output-dir runs/search
```

`--n-trials` is a maximum: the search stops automatically when the
termination criterion is met (checked from `--stop-min-trials`, default 20).
Use `--stop-threshold 0.5` to set a fixed regret tolerance in objective
units, or `--no-early-stop` to always run the full budget.

**3. Retrain from a known configuration** (no search)

```bash
python -m toolwear train \
    --config runs/search/best_config.json \
    --sensor-features features/features_sensor_train_K5.csv \
    --controller-features features/features_controller_train.csv \
    --labels Training_Dataset/trainset_toolwear_measurement.csv \
    --output-dir runs/model
```

The sensor features file must match the `K` stored in the configuration.

**4. Predict on new data**

```bash
python -m toolwear predict \
    --model-dir runs/model \
    --sensor-features features/features_sensor_eval_K5.csv \
    --controller-features features/features_controller_eval.csv \
    --labels Evaluation_Dataset/evalset_toolwear_measurement.csv \
    --output-dir runs/predictions
```

`--labels` is optional. When given, the cut-1 measurement anchors each
prediction and the remaining cuts are scored (`metrics.json`,
`predictions.png`). Without it, the training median is used as the anchor.

### Python API

```python
import pandas as pd
from toolwear import ModelConfig, ToolWearModel
from toolwear.data import load_labels

model = ToolWearModel(ModelConfig.load("best_config.json")).fit(
    pd.read_csv("features/features_sensor_train_K5.csv"),
    load_labels("trainset_toolwear_measurement.csv"),
    pd.read_csv("features/features_controller_train.csv"),
)
predictions = model.predict(sensor_df, controller_df, initial_wear={1: 61.2, 2: 58.9})
```

## Search outputs

| File | Content |
|------|---------|
| `best_config.json` | Complete configuration of the best trial |
| `model/` | Fitted model ready for `predict` |
| `trials_summary.csv` | Every completed trial with CV diagnostics |
| `outer_convergence.{png,json}` | Best score vs. trial and a plateau diagnosis |
| `termination_history.csv`, `termination.png` | Regret bound vs. stopping threshold per trial |
| `optuna_{slice,rank,parallel_coordinate}_{K,lambda_loss}.png` | Effect of K and λ |
| `encoders/K{K}/lambda_{λ}/` | Cached encoders and their inner-search convergence |
| `optuna.db` | Optuna storage for both searches (enables resuming) |

