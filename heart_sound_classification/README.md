# Heart Sound Classification

This project classifies heart sound recordings using a machine learning pipeline built in the notebook [heart.ipynb](heart.ipynb). The goal is to take a heart sound recording, extract compact audio features, and predict the corresponding heart sound class.

The notebook is a complete end-to-end workflow:

- load label metadata from the dataset CSV
- match each record to its `.wav` file
- extract audio features with `librosa`
- train and evaluate a classification model
- run inference on a single audio file

## Dataset

The data comes from the UCI HLS-CMDS collection:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

The notebook uses the heart sound portion of the dataset:

- `dataset/HS.csv` for labels and metadata
- `dataset/HS/` for the corresponding `.wav` files

Each `Heart Sound ID` in the CSV maps to a matching audio file.

The dataset also includes lung sound and mixed sound folders, but the notebook focuses on the heart sound subset.

## What the notebook does

The workflow in [heart.ipynb](heart.ipynb) is:

1. Load the metadata from `HS.csv`.
2. Inspect class distribution and remove classes with too few samples.
3. Build file paths for each heart sound recording from `Heart Sound ID`.
4. Load each audio file at a fixed sample rate of 22050 Hz.
5. Extract audio features from each `.wav` file:
   - 20 MFCC coefficients
   - zero crossing rate
   - RMS energy
   - spectral centroid
   - spectral bandwidth
6. Combine those features into a 24-dimensional feature vector per recording.
7. Split the data with stratification so class proportions are preserved.
8. Train a `RandomForestClassifier` inside a preprocessing pipeline with `StandardScaler`.
9. Evaluate the model with a train/test split, classification report, confusion matrix, and 5-fold stratified cross-validation.
10. Retrain on the full dataset and provide a helper function for single-file prediction.

## Model Summary

- Feature extraction: `librosa`
- Classifier: `RandomForestClassifier`
- Trees: `300`
- Class balance: `balanced`
- Validation: stratified train/test split plus 5-fold cross-validation
- Input feature size: 24 values per sample

## Method Details

The feature vector is built from both time-domain and frequency-domain summaries.

- MFCCs capture the shape of the audio spectrum over time.
- Zero crossing rate helps describe waveform noisiness and signal changes.
- RMS energy provides a measure of signal strength.
- Spectral centroid and spectral bandwidth summarize where the energy is concentrated in the spectrum.

These features are averaged across the full audio clip, which keeps the model simple and makes it suitable for tabular machine learning methods such as random forests.

## Repository Structure

- `heart_sound_classification/heart.ipynb` - main notebook
- `dataset/HS.csv` - label and metadata file
- `dataset/HS/` - heart sound recordings
- `dataset/LS/` - lung sound recordings
- `dataset/Mix/` - mixed audio recordings

## Environment Setup

The notebook expects a Python environment with the dependencies used in the import cells.

Recommended packages:

- `pandas`
- `numpy`
- `librosa`
- `matplotlib`
- `scikit-learn`

If you are setting this up from scratch, install the packages in your environment before opening the notebook.

## Requirements

The notebook uses Python packages including:

- `pandas`
- `numpy`
- `librosa`
- `matplotlib`
- `scikit-learn`

## Usage

1. Open [heart.ipynb](heart.ipynb) in Jupyter or VS Code.
2. Make sure the workspace paths still point to `../dataset` from the notebook location.
3. Run the cells from top to bottom.

The notebook will print the class distribution, the number of valid samples loaded, the feature matrix shape, evaluation metrics, and cross-validation scores.

For a single prediction, the notebook defines:

```python
predict_heart(file_path)
```

which loads an audio file, extracts features, and returns the predicted heart sound class.

Example:

```python
predict_heart("../dataset/HS/F_N_RC.wav")
```

## Notes

- The notebook expects relative paths to the dataset folder shown above.
- Some classes are removed if they have fewer than 5 samples, so the final label set may be smaller than the raw CSV label list.
- Missing or unreadable audio files are skipped during feature extraction.
- The final model is trained on the full dataset after evaluation, so the helper prediction function uses the most recently fitted model.