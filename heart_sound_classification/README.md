# Heart Sound Classification

This folder contains the heart sound classification workflow implemented in [heart.ipynb](heart.ipynb). The notebook builds several classical machine learning models on the heart-sound subset of HLS-CMDS, compares them with the same feature representation, and saves the best-performing model for later reuse.

## Dataset

The notebook uses the heart subset of the UCI HLS-CMDS collection:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

Inputs used by the notebook:

- `dataset/HS.csv` for labels and metadata
- `dataset/HS/` for the corresponding `.wav` files

Each `Heart Sound ID` in the CSV maps to one audio file in `dataset/HS/`.

## Workflow

The notebook follows an end-to-end pipeline:

1. Load `HS.csv` into a DataFrame and inspect the class distribution.
2. Remove classes with fewer than 5 samples to reduce extreme imbalance.
3. Build absolute file paths from `Heart Sound ID` values.
4. Load each recording at 22050 Hz with `librosa`.
5. Extract a fixed-length feature vector from each file.
6. Encode labels and persist the encoder as `label_encoder_heart.pkl`.
7. Split the data with `train_test_split(..., stratify=y)`.
8. Train and compare multiple models.
9. Evaluate each model with accuracy, macro F1, classification report, confusion matrix, and 5-fold stratified cross-validation.
10. Select the best model by macro F1, retrain on all available data, and save it as `heart_model.pkl`.
11. Provide a helper for single-file inference through `predict_heart(file_path)`.

## Architecture

The notebook uses a compact audio-feature pipeline followed by classical classifiers.

### Feature extraction layer

Each recording is converted into a 64-dimensional feature vector:

- 20 MFCC means
- 20 MFCC standard deviations
- 12 chroma features
- 7 spectral contrast values
- zero-crossing rate
- RMS energy
- spectral centroid
- spectral bandwidth
- spectral rolloff

These statistics are pooled over the full clip, which keeps the representation fixed-length and suitable for tabular models.

### Modeling layer

The notebook compares three models, all wrapped in a preprocessing pipeline with `StandardScaler`:

- `DummyClassifier(strategy="most_frequent")` as a baseline
- `RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)`
- `SVC(kernel="rbf", C=10, class_weight="balanced")`
- `XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)`

The final model is whichever achieves the best mean macro F1 score during cross-validation.

### Validation layer

Validation uses two levels of checking:

- an 80/20 stratified train/test split for quick evaluation
- 5-fold `StratifiedKFold` cross-validation for a more stable model comparison

## Files

- `heart.ipynb` - main training and inference notebook
- `heart_model.pkl` - saved best model
- `label_encoder_heart.pkl` - saved label encoder

## Usage

1. Open [heart.ipynb](heart.ipynb) in Jupyter or VS Code.
2. Confirm that the notebook can resolve `../dataset/HS.csv` and `../dataset/HS/`.
3. Run the cells from top to bottom.

The notebook prints the filtered class distribution, feature matrix shape, evaluation metrics, cross-validation scores, and the selected best model.

For inference, the notebook exposes:

```python
predict_heart(file_path)
```

Example:

```python
predict_heart("../dataset/HS/F_N_RC.wav")
```

## Notes

- Classes with fewer than 5 samples are removed before training.
- Missing or unreadable files are skipped during feature extraction.
- The saved model and label encoder are required if you want to reuse the trained pipeline outside the notebook.

## Citation

If you use this dataset or this notebook, please cite the HLS-CMDS dataset descriptor:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.