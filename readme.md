# Few-Shot Tool Wear Prediction Using Self-Supervised Learning and Multimodal Feature Fusion

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
   stops early following Makarova et al. (2022), using Optuna's
   `RegretBoundEvaluator`. The search stops once the high-probability bound on
   the remaining regret falls below the statistical error of the best trial's
   robust CV score. Beyond that point, further gains in CV score are
   within the estimation error and are unlikely to improve generalization.

## Repository layout

```
scripts/
  extract_controller_features.py   # controller features per checkpoint
  extract_sensor_features.py       # windowed sensor features for one or more K
  (data_loader_trainset.py)        # official loader, to be copied here (see Data)
  (data_loader_evalset.py)         # official loader, to be copied here (see Data)
toolwear/
  __main__.py       # entry point for `python -m toolwear`
  cli.py            # command-line interface (tune / train / predict)
  config.py         # configuration dataclasses (serializable to JSON)
  data.py           # loading and shaping of features and labels
  encoder.py        # causal TCN, SSL losses and training, TCNEncoder
  regressor.py      # SVR pipeline, autoregressive prediction, LODO CV
  model.py          # ToolWearModel: fit / predict / save / load
  tuning.py         # nested Optuna search (Tuner)
  termination.py    # automatic termination criterion for the outer search
  reporting.py      # metrics, plots and search diagnostics
  utils.py          # reproducibility (seeding, determinism) and device helpers
features/           # precomputed features used in our experiments (see Usage)
  features_sensor_{train,eval}_K{1,5,10,20}.csv
  features_controller_{train,eval}.csv
runs/               # outputs of our experiments
  search/           # complete hyperparameter search (functionality 3, see Search outputs)
    best_config.json
    model/          # final model selected by the search
    ...
  model/            # model retrained from best_config.json (functionality 2)
  predictions/      # predictions on the evaluation set (functionality 1)
requirements.txt
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
   them from. They are only needed to recompute the features.

## Installation

The experiments were run with Python 3.9.

```bash
pip install -r requirements.txt
```

The fitted regressor is stored as a pickled scikit-learn object, so
`requirements.txt` pins scikit-learn (1.6.1), together with NumPy, SciPy and
PyWavelets, to the versions used to train it. Loading the included models with
other versions may fail or emit warnings.

## Usage

### Reproducibility note

The hyperparameter search is not bit-for-bit reproducible across machines:
although all seeds are fixed, the selected configuration may vary slightly
with the hardware (CPU or GPU), the numerical libraries and their versions.
For this reason, `runs/` contains the complete search obtained in our
experiments together with the final model and its predictions, so the
reported results can be reproduced without repeating the search.

All the functionalities below use the features extracted from the raw data.
Since their computation is slow, the feature files used in our experiments are
included in `features/` for convenience, and the commands below already point
to them. The label files (`*_toolwear_measurement.csv`) are still read from the
downloaded datasets (see Data).

The features can be recomputed from the raw data with the following commands
(this overwrites the files in `features/`; use another `--output-dir` to keep
them):

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

When recomputing, functionalities 1 and 2 only need `-k 10`, the number of
windows of the selected configuration; the full search (functionality 3) needs
every value.

### Functionality

The code provides three functionalities, from the quickest to the most complete:

1. **Predict with the selected model**: applies the final model of our
   experiments to the evaluation data, reproducing the reported results.
2. **Retrain the model from the included search**: trains the encoder and the
   regressor again with the configuration selected in our search, without
   repeating the search.
3. **Run the complete experiment**: repeats the whole nested hyperparameter
   search and fits the model it selects.

The outputs of the three functionalities obtained in our experiments are
included in `runs/`:

| Folder | Functionality | Content |
|--------|---------------|---------|
| `runs/search/` | 3 | Complete search: Optuna storage, cached encoders, trial summary, convergence and termination reports, `best_config.json` and the selected model (`model/`) |
| `runs/model/` | 2 | Model retrained from `runs/search/best_config.json` |
| `runs/predictions/` | 1 | Predictions of `runs/search/model` on the evaluation set and their metrics |

The commands below write to new folders (`runs/model_retrained`,
`runs/search_new`) so that the included results are not overwritten.

### 1. Predict with the selected model

```bash
python -m toolwear predict \
    --model-dir runs/search/model \
    --sensor-features features/features_sensor_eval_K10.csv \
    --controller-features features/features_controller_eval.csv \
    --labels Evaluation_Dataset/evalset_toolwear_measurement.csv \
    --output-dir runs/predictions
```

This writes `predictions.csv` and, when `--labels` is given, `metrics.json`
and `predictions.png`. The included `runs/predictions/metrics.json` contains
the results reported in the paper on the evaluation set:

| RMSE | MAPE | R² |
|------|------|----|
| 9.226 | 7.84 % | 0.919 |

`--labels` is optional. When given, the cut-1 measurement anchors each
prediction and the remaining cuts are scored; `predictions.csv` then also
contains the measurements and the errors. Without it, the training median is
used as the anchor and only the predictions are written.

### 2. Retrain the model from the included search

To train the encoder and the regressor again with the configuration selected
in our search, without repeating the search:

```bash
python -m toolwear train \
    --config runs/search/best_config.json \
    --sensor-features features/features_sensor_train_K10.csv \
    --controller-features features/features_controller_train.csv \
    --labels Training_Dataset/trainset_toolwear_measurement.csv \
    --output-dir runs/model_retrained
```

The sensor features file must match the `K` stored in the configuration.
Predictions are then obtained as in functionality 1 with
`--model-dir runs/model_retrained`. The result of this step in our experiments
is `runs/model/`.

### 3. Run the complete experiment

To repeat the whole procedure, run the hyperparameter search on the features
extracted above (resumable; writes `best_config.json` and a fitted model):

```bash
python -m toolwear tune \
    --sensor-template "features/features_sensor_train_K{k}.csv" \
    --controller-features features/features_controller_train.csv \
    --labels Training_Dataset/trainset_toolwear_measurement.csv \
    --output-dir runs/search_new
```

`--n-trials` is a maximum: the search stops automatically when the
termination criterion is met (checked from `--stop-min-trials`, default 20).
Use `--stop-threshold 0.5` to set a fixed regret tolerance in objective
units, or `--no-early-stop` to always run the full budget. Predictions are
then obtained as in functionality 1 with `--model-dir runs/search_new/model`.

Pointing `--output-dir` to `runs/search` resumes the included search from its
Optuna storage instead of starting a new one.

## Search outputs

Contents of the search folder (`runs/search/` for our experiments):

| File | Content |
|------|---------|
| `best_config.json` | Complete configuration of the best trial |
| `model/` | Fitted model ready for `predict` (`model.json`, `encoder/`, `regressor.joblib`) |
| `trials_summary.csv` | Every completed trial with CV diagnostics |
| `outer_convergence.{png,json}`, `outer_running_best.csv` | Best score vs. trial and a plateau diagnosis |
| `termination_history.csv`, `termination.png` | Regret bound vs. stopping threshold per trial |
| `encoders/K{K}/lambda_{λ}/` | Cached encoders (`encoder.json`, `weights.pt`, `preprocessor.joblib`) and the inner-search convergence (`inner_convergence.json`, `inner_running_best.csv`) |
| `optuna.db` | Optuna storage for both searches (enables resuming) |
