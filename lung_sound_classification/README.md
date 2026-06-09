# Lung Sound Classification

This document describes the `lung.ipynb` workflow and implementation details for lung sound classification using the HLS-CMDS dataset.

## Summary

The notebook builds a compact, reproducible pipeline that converts WAV recordings into a fixed-length feature vector and trains several classical classifiers. It selects the best model by macro F1 using 5-fold stratified cross-validation, retrains it on all data, and saves the model and label encoder for later inference and evaluation.

## Data

- Source: UCI HLS-CMDS (Heart and Lung Sounds Dataset)
- Metadata: `dataset/LS.csv`
- Audio files: `dataset/LS/` (WAV)

Each `Lung Sound ID` in the CSV corresponds to a WAV file in `dataset/LS/`.

## End-to-end Workflow

1. Load `dataset/LS.csv` and inspect class counts.
2. Filter out classes with fewer than 5 samples (configurable threshold) to reduce noise from extremely small classes.
3. Construct file paths and verify file existence.
4. Extract features from each WAV at 22050 Hz using `librosa`.
5. Encode labels with `LabelEncoder` and persist it to `label_encoder_lung.pkl`.
6. Split data with `train_test_split(..., stratify=y)` for an 80/20 evaluation.
7. Train candidate models inside `Pipeline` objects that include `StandardScaler`.
8. Evaluate models using stratified 5-fold cross-validation (macro F1) and the hold-out test set (accuracy, macro F1, classification report, confusion matrix).
9. Select the best-performing model from CV, retrain on the full dataset, and save it as `lung_model.pkl`.
10. Provide `predict_lung(file_path)` for single-file inference using the saved artifact.

## Feature Representation

Each recording is summarized into a fixed-length vector composed of the following pooled statistics:

- 20 MFCC means
- 20 MFCC standard deviations
- 12 chroma means
- 7 spectral contrast means
- zero-crossing rate (mean)
- RMS energy (mean)
- spectral centroid (mean)
- spectral bandwidth (mean)
- spectral rolloff (mean)

Total feature length: 64 elements. These are simple, fast to compute, and suitable for tree- and kernel-based models.

Rationale: mean/std pooling produces a compact representation invariant to recording length, which simplifies model training and reduces compute requirements compared with frame-level or spectrogram models.

## Models Compared

All candidates are trained inside a `Pipeline([('scaler', StandardScaler()), ('model', ...)])`:

- Baseline: `DummyClassifier(strategy='most_frequent')`
- Random Forest: `RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)`
- SVM: `SVC(kernel='rbf', C=10, class_weight='balanced')`
- XGBoost: `XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42)`

Selection metric: mean macro F1 across 5 stratified folds. The notebook also reports hold-out test metrics and confusion matrices per model.

## Outputs and Artifacts

- `lung_model.pkl` — the selected model retrained on all available data
- `label_encoder_lung.pkl` — label encoder used to map between numeric and string labels
- Printed artifacts in the notebook: classification reports, confusion matrices, CV scores

Saved artifacts are written to the `lung_sound_classification/` folder by default.

## How to run

1. Activate the project virtual environment and install dependencies (see top-level `requirements.txt`):

```bash
source ../venv/bin/activate
pip install -r ../requirements.txt
```

2. Open `lung.ipynb` in Jupyter or VS Code and run cells from top to bottom.

3. To reuse the trained model programmatically:

```python
import joblib
model = joblib.load('lung_model.pkl')
le = joblib.load('label_encoder_lung.pkl')
# extract features with the same extract_features(path) used in the notebook
```

## Limitations & Next Steps

- Pooling statistics remove temporal structure; for overlapping sources or complex temporal patterns, consider frame-based models or CNNs on mel-spectrograms.
- No explicit source separation is performed; mixed recordings will likely reduce single-task performance.
- For production use, add input validation, unit tests for the feature extractor, and deterministic dataset splits saved as manifests.

Suggested improvements:
- Replace pooled features with short-time frame stacks or mel-spectrogram inputs to a CNN.
- Add balanced resampling or class-specific augmentation for rare classes.
- Provide an easy CLI or `predict.py` wrapper for batch inference.

## Citation

When using this dataset or the notebooks, please cite:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.



