# Heart & Lung Sound Classification (HLS-CMDS)

Overview
--------

This project builds machine learning models to classify heart and lung sounds using the HLS-CMDS dataset (clinical manikin recordings from UCI). It includes full pipelines for feature extraction, model training, evaluation, and mixed-signal inference.

The system demonstrates a complete audio ML workflow using classical machine learning techniques.

Dataset
-------

The data used in this project comes from the UCI HLS-CMDS collection:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

Please cite the dataset descriptor when using the data or these notebooks:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.


Key features
-------

- Heart sound classification model (Random Forest)
- Lung sound classification model (Random Forest)
- Mixed audio evaluation pipeline (heart + lung prediction)
- Audio feature extraction using librosa
- Stratified train-test split + cross-validation
- Confusion matrix and performance reporting
- Single-file inference support

Methodology
-------

Each audio file is processed using librosa and converted into a fixed-length feature vector:

- 20 MFCC features (mean pooled)
- Zero Crossing Rate
- Root Mean Square Energy
- Spectral Centroid
- Spectral Bandwidth

These features are used to train a Random Forest classifier with class balancing.

Repository structure
--------------------

- `heart_sound_classification/` — notebook and assets for heart sound modeling
- `lung_sound_classification/` — notebook and assets for lung sound modeling
- `mix_sound_classification/` — notebook to evaluate models on mixed recordings
- `dataset/` — CSV metadata (`HS.csv`, `LS.csv`, `Mix.csv`) and subfolders for WAV files (`HS/`, `LS/`, `Mix/`)
- `requirements.txt` — project dependencies

Notebooks and workflow
----------------------

- `heart_sound_classification/heart.ipynb` —
  - Loads `dataset/HS.csv`, filters small classes, maps `Heart Sound ID` to `dataset/HS/*.wav`.
  - Extracts per-file features: 20 MFCC means, zero-crossing rate (mean), RMS (mean), spectral centroid (mean), spectral bandwidth (mean).
  - Builds a 24-dimensional feature vector per recording, trains a `RandomForestClassifier` inside a `Pipeline` with `StandardScaler`, evaluates with stratified train/test split and 5-fold cross-validation, and exposes `predict_heart(file_path)`.

- `lung_sound_classification/lung.ipynb` — same pipeline for lung sounds (uses `dataset/LS.csv` and `dataset/LS/*.wav`) and provides `predict_lung(file_path)`.

- `mix_sound_classification/mix.ipynb` — loads trained heart and lung models, extracts features from mixed recordings (`dataset/Mix.csv` -> `dataset/Mix/*.wav`), runs both models on each mixed file, saves aggregated results (`mix_evaluation_results.csv`), and shows per-task evaluation metrics and confusion matrices.

Requirements and setup
----------------------

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Files you may want to create / save
----------------------------------

- Save trained models for reuse (the `mix` notebook expects `heart_model.pkl` and `lung_model.pkl` in the respective folders). Use `joblib.dump(model, "heart_model.pkl")` or similar.
- Optionally add `requirements.txt` version pins or an `environment.yml` for reproducibility.

Running the notebooks
---------------------

Open each notebook in Jupyter or VS Code and execute cells sequentially. The notebooks print class distributions, loaded sample counts, feature matrix shapes, evaluation metrics, and cross-validation results.

Single-file inference examples
-----------------------------

In `heart.ipynb`:

```python
predict_heart("../dataset/HS/F_N_RC.wav")
```

In `lung.ipynb`:

```python
predict_lung("../dataset/LS/F_PR_LLA.wav")
```

Notes, limitations, and future directions
-----------------------------------------

- Current models use per-file aggregated features (means of MFCCs and basic spectral/time statistics). This approach is simple and fast, but it discards temporal structure inside recordings.
- Class imbalance is handled by `class_weight="balanced"` in the Random Forest, and classes with fewer than 5 samples are filtered out.
- Missing or unreadable WAV files are skipped during feature extraction; check console output for "Missing:" messages.

Future enhancements (suggested)
------------------------------

- Replace per-file aggregated features with frame-level representations or spectrogram images and train convolutional neural networks (CNNs) for improved performance on complex patterns.
- Add model serialization and a small CLI or REST API for batch prediction.
- Add unit tests for feature extraction and model inference pipelines.

Citation
--------

If you use this dataset or the notebooks, please cite:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.
