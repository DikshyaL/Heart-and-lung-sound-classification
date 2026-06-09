# Lung Sound Classification

This folder contains the lung sound classification workflow implemented in [lung.ipynb](lung.ipynb). The notebook uses the lung subset of HLS-CMDS, compares multiple classical ML models on the same audio representation, and saves the best model for reuse.

## Dataset

The notebook uses the lung subset of the UCI HLS-CMDS collection:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

Inputs used by the notebook:

- `dataset/LS.csv` for labels and metadata
- `dataset/LS/` for the corresponding `.wav` files

Each `Lung Sound ID` in the CSV maps to one audio file in `dataset/LS/`.

## Workflow

The notebook follows an end-to-end pipeline:

1. Load `LS.csv` into a DataFrame and inspect the label distribution.
2. Remove classes with fewer than 5 samples to reduce imbalance noise.
3. Build file paths from `Lung Sound ID` values.
4. Load each recording at 22050 Hz with `librosa`.
5. Extract a fixed-length feature vector from each file.
6. Encode labels and persist the encoder as `label_encoder_lung.pkl`.
7. Split the data with `train_test_split(..., stratify=y)`.
8. Train and compare several models.
9. Evaluate each model with accuracy, macro F1, classification report, confusion matrix, and 5-fold stratified cross-validation.
10. Select the best model by macro F1, retrain on all available data, and save it as `lung_model.pkl`.
11. Provide a helper for single-file inference through `predict_lung(file_path)`.

## Architecture

The lung notebook uses the same feature and model stack as the heart notebook so the two tasks can be compared fairly.

### Feature extraction layer

Each audio file is converted into a 64-dimensional feature vector:

- 20 MFCC means
- 20 MFCC standard deviations
- 12 chroma features
- 7 spectral contrast values
- zero-crossing rate
- RMS energy
- spectral centroid
- spectral bandwidth
- spectral rolloff

The features are pooled across the full recording, which gives a compact tabular representation.

### Modeling layer

All models are wrapped in a preprocessing pipeline with `StandardScaler`:

- `DummyClassifier(strategy="most_frequent")` as a baseline
- `RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)`
- `SVC(kernel="rbf", C=10, class_weight="balanced")`
- `XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)`

The notebook selects the model with the best mean macro F1 score across cross-validation folds.

### Validation layer

Validation is done with:

- an 80/20 stratified train/test split
- 5-fold `StratifiedKFold` cross-validation

## Files

- `lung.ipynb` - main training and inference notebook
- `lung_model.pkl` - saved best model
- `label_encoder_lung.pkl` - saved label encoder

## Usage

1. Open [lung.ipynb](lung.ipynb) in Jupyter or VS Code.
2. Confirm that the notebook can resolve `../dataset/LS.csv` and `../dataset/LS/`.
3. Run the cells from top to bottom.

The notebook prints filtered class counts, feature matrix shape, evaluation metrics, cross-validation scores, and the selected best model.

For inference, the notebook exposes:

```python
predict_lung(file_path)
```

Example:

```python
predict_lung("../dataset/LS/F_PR_LLA.wav")
```

## Notes

- Classes with fewer than 5 samples are removed before training.
- Missing or unreadable files are skipped during feature extraction.
- The saved model and label encoder are required if you want to reuse the trained pipeline outside the notebook.

## Citation

If you use this dataset or this notebook, please cite the HLS-CMDS dataset descriptor:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.


