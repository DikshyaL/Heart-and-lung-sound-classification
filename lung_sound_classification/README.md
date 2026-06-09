# Lung Sound Classification

This document describes the `lung.ipynb` workflow and implementation details for lung sound classification using the HLS-CMDS dataset.

## Summary

The notebook builds a compact, reproducible pipeline that converts WAV recordings into a fixed-length feature vector and trains several classical classifiers. It selects the best model by macro F1 using 5-fold stratified cross-validation, retrains it on all data, and saves the model and label encoder for later inference and evaluation.

## Data


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


Total feature length: 64 elements. These are simple, fast to compute, and suitable for tree- and kernel-based models.

Rationale: mean/std pooling produces a compact representation invariant to recording length, which simplifies model training and reduces compute requirements compared with frame-level or spectrogram models.

## Models Compared

All candidates are trained inside a `Pipeline([('scaler', StandardScaler()), ('model', ...)])`:


Selection metric: mean macro F1 across 5 stratified folds. The notebook also reports hold-out test metrics and confusion matrices per model.

## Outputs and Artifacts


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


Suggested improvements:

## Citation

When using this dataset or the notebooks, please cite:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.



