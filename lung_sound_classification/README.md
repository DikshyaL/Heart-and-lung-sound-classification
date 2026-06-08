# Lung Sound Classification

This project classifies lung sound recordings using a machine learning pipeline implemented in the notebook [lung.ipynb](lung.ipynb). The goal is to extract compact audio features from `.wav` recordings and predict the lung sound class using a tabular ML model.

## Dataset

The data comes from the UCI HLS-CMDS collection:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

The notebook focuses on the lung-sound subset:

- `dataset/LS.csv` — labels and metadata
- `dataset/LS/` — `.wav` audio recordings

Each `Lung Sound ID` in the CSV maps to a corresponding `.wav` file in `dataset/LS/`.

## Notebook workflow

The pipeline in [lung.ipynb](lung.ipynb) follows these steps:

1. Load `dataset/LS.csv` into a pandas DataFrame and inspect label distribution.
2. Filter out classes with very few samples (the notebook keeps classes with at least 5 examples).
3. Construct relative file paths from `Lung Sound ID` values and verify files exist.
4. Load each `.wav` at 22050 Hz and extract summary audio features:
   - 20 MFCC coefficients (mean across frames)
   - zero crossing rate (mean)
   - RMS energy (mean)
   - spectral centroid (mean)
   - spectral bandwidth (mean)
5. Concatenate the per-file summaries into a fixed-length feature vector (24 features) and build X/y arrays.
6. Split the data with `train_test_split(..., stratify=y)` to preserve class proportions.
7. Train a `RandomForestClassifier` within a `Pipeline` that applies `StandardScaler`.
8. Evaluate with accuracy, classification report, confusion matrix visualization, and 5-fold stratified cross-validation.
9. Retrain the final model on the full dataset and expose a helper function for single-file inference.

## Modeling details

- Feature extraction library: `librosa`
- Classifier: `RandomForestClassifier` (n_estimators=300)
- Class weighting: `balanced` to mitigate class imbalance
- Cross-validation: `StratifiedKFold(n_splits=5)`
- Final model: retrained on all available (filtered) samples for inference

## Feature engineering rationale

This project uses simple aggregated statistics (means) of spectral and temporal features to create a compact, fixed-size input for classical ML models. Advantages:

- Fast to compute and robust to varying recording lengths
- Works well with tree-based models without heavy normalization

Limitations:

- Averaging removes temporal structure — models cannot leverage patterns across time windows
- More advanced approaches (CNNs on spectrograms or sequence models on frames) may improve performance for complex tasks

## Files and structure

- `lung_sound_classification/lung.ipynb` — main notebook implementing the pipeline
- `dataset/LS.csv` — metadata and labels for lung sounds
- `dataset/LS/` — WAV files for lung recordings
- `dataset/HS/`, `dataset/Mix/` — other dataset subsets (heart/mixed) included in the original collection

## Environment and requirements

Recommended Python packages (used in the notebook):

- pandas
- numpy
- librosa
- matplotlib
- scikit-learn

Install quickly into an active virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas numpy librosa matplotlib scikit-learn
```

## Usage

1. Ensure the `dataset` folder (with `LS.csv` and `LS/` wav files) is at the workspace root relative to the notebook location.
2. Open [lung.ipynb](lung.ipynb) and run cells sequentially.
3. The notebook prints dataset stats, sample counts, feature shapes, evaluation metrics, and cross-validation scores.

Single-file inference

The notebook defines a helper function `predict_lung(file_path)` which:

- loads the file at 22050 Hz
- computes the same 24-dimensional feature vector
- returns the predicted lung sound class using the fitted model

Example:

```python
predict_lung("../dataset/LS/F_PR_LLA.wav")
```

## Notes and troubleshooting

- The notebook removes classes with fewer than 5 samples; your label set may differ from the raw CSV.
- Missing or unreadable WAV files are skipped — check printed "Missing" messages if you expect files to load.
- Feature extraction uses `librosa.load` which can be sensitive to file encodings and sample rates; convert files to WAV PCM if you encounter errors.

## Citation

If you use this dataset or the notebooks in this repository, please cite the dataset descriptor:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.


