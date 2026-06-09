# Heart & Lung Sound Classification (HLS-CMDS)

This repository contains three notebooks that share a common audio feature pipeline:

- [heart_sound_classification/heart.ipynb](heart_sound_classification/heart.ipynb) trains and evaluates heart sound classifiers.
- [lung_sound_classification/lung.ipynb](lung_sound_classification/lung.ipynb) trains and evaluates lung sound classifiers.
- [mix_sound_classification/mix.ipynb](mix_sound_classification/mix.ipynb) loads the saved models and evaluates them on mixed recordings.

Together, they form a complete audio machine learning workflow: dataset loading, feature extraction, model comparison, validation, artifact saving, and mixed-signal inference.

## Dataset

All notebooks use the UCI HLS-CMDS collection:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

Dataset inputs in this repository:

- `dataset/HS.csv` and `dataset/HS/` for heart sounds
- `dataset/LS.csv` and `dataset/LS/` for lung sounds
- `dataset/Mix.csv` and `dataset/Mix/` for mixed recordings

## Project Architecture

The project uses a shared classical ML architecture for the heart and lung tasks, then reuses the saved models for mixed-recording evaluation.

### Shared feature extraction

Each audio file is loaded at 22050 Hz and converted into a 64-dimensional summary vector:

- 20 MFCC means
- 20 MFCC standard deviations
- 12 chroma features
- 7 spectral contrast values
- zero-crossing rate
- RMS energy
- spectral centroid
- spectral bandwidth
- spectral rolloff

This representation is compact, fixed length, and compatible with tabular machine learning models.

### Heart and lung model stack

The heart and lung notebooks both compare the same family of models:

- `DummyClassifier(strategy="most_frequent")` as a baseline
- `RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)`
- `SVC(kernel="rbf", C=10, class_weight="balanced")`
- `XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)`

Each model is wrapped in a `Pipeline` with `StandardScaler` before training. The best model is selected by mean macro F1 score from 5-fold stratified cross-validation, retrained on all available samples, and saved with `joblib`.

### Mixed-recording evaluation layer

The mix notebook does not train a new model. Instead, it:

- loads `heart_model.pkl` and `lung_model.pkl`
- loads the corresponding label encoders
- extracts the same shared features from mixed audio files
- predicts heart and lung labels from the same input vector
- compares predictions to the labels in `Mix.csv`

## Notebook Workflows

### Heart notebook

- Loads `HS.csv`, filters classes with fewer than 5 samples, and maps `Heart Sound ID` values to `dataset/HS/*.wav`.
- Builds the feature matrix and encodes labels.
- Trains and compares the baseline, Random Forest, SVM, and XGBoost pipelines.
- Reports accuracy, macro F1, classification reports, confusion matrices, and 5-fold cross-validation scores.
- Saves `heart_model.pkl` and `label_encoder_heart.pkl`.

### Lung notebook

- Loads `LS.csv`, filters classes with fewer than 5 samples, and maps `Lung Sound ID` values to `dataset/LS/*.wav`.
- Reuses the same feature extraction and model comparison pipeline as the heart notebook.
- Reports the same evaluation metrics and saves `lung_model.pkl` plus `label_encoder_lung.pkl`.

### Mix notebook

- Loads the saved heart and lung models plus their encoders.
- Iterates through `Mix.csv` and predicts both tasks for every mixed recording.
- Saves a per-file evaluation table to `mix_evaluation_results.csv`.
- Produces classification reports and confusion matrices for both tasks.

## Requirements and Setup

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## How to Run

1. Run the heart notebook first so `heart_model.pkl` and `label_encoder_heart.pkl` are available.
2. Run the lung notebook so `lung_model.pkl` and `label_encoder_lung.pkl` are available.
3. Run the mix notebook to evaluate both saved models on mixed recordings.
4. Open each notebook in Jupyter or VS Code and execute cells sequentially.

The notebooks print filtered class distributions, sample counts, feature shapes, evaluation metrics, and cross-validation results.

## Outputs

- `heart_sound_classification/heart_model.pkl`
- `heart_sound_classification/label_encoder_heart.pkl`
- `lung_sound_classification/lung_model.pkl`
- `lung_sound_classification/label_encoder_lung.pkl`
- `mix_sound_classification/mix_evaluation_results.csv`

## Limitations and Future Work

- The current feature set is a single global summary for each recording, so temporal structure is discarded.
- Class imbalance is reduced by filtering very small classes and using `class_weight="balanced"` in the tree and SVM models.
- Mixed recordings are harder to classify because the heart and lung sources overlap in the same signal.
- A future upgrade could replace aggregated features with spectrograms or mel-spectrograms and train CNNs for better source-aware classification.

## Citation

If you use this dataset or the notebooks in this repository, please cite the HLS-CMDS dataset descriptor:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.
