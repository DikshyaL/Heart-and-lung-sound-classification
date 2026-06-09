# Heart Sound Classification

This folder contains the heart-sound classification workflow implemented in [heart.ipynb](heart.ipynb). The notebook trains classical machine learning models on the isolated heart subset of HLS-CMDS, compares them with a shared feature representation, and saves the best-performing artifact for reuse.

## Project Summary

The heart notebook investigates classification of isolated heart sound recordings using traditional machine learning techniques. It is designed to establish a baseline model that can later be evaluated against mixed heart-lung recordings in the top-level project workflow.

## Dataset

The notebook uses the heart subset of the HLS-CMDS dataset:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

Inputs used by the notebook:

- `dataset/HS.csv` for labels and metadata
- `dataset/HS/` for the corresponding WAV files

Each `Heart Sound ID` in the CSV maps to one audio file in `dataset/HS/`.

## Feature Extraction

Each recording is converted into a fixed-length 64-dimensional feature vector:

- 20 MFCC means
- 20 MFCC standard deviations
- 12 chroma features
- 7 spectral contrast values
- zero-crossing rate
- RMS energy
- spectral centroid
- spectral bandwidth
- spectral rolloff

These statistics are pooled over the full clip, which keeps the representation compact and suitable for tabular classifiers.

## Machine Learning Models

The notebook compares the following models inside a preprocessing pipeline with `StandardScaler`:

- Dummy Classifier as a baseline
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

The final model is selected by mean macro F1 score during stratified cross-validation.

## Evaluation Metrics

Model performance is reported using:

- Accuracy
- Macro F1 score
- Classification report
- Confusion matrix
- Stratified K-fold cross validation

## Workflow

1. Load `HS.csv` into a DataFrame and inspect the class distribution.
2. Remove classes with fewer than 5 samples to reduce extreme imbalance.
3. Build absolute file paths from `Heart Sound ID` values.
4. Load each recording at 22050 Hz with `librosa`.
5. Extract the shared fixed-length feature vector.
6. Encode labels and persist the encoder as `label_encoder_heart.pkl`.
7. Split the data with `train_test_split(..., stratify=y)`.
8. Train and compare multiple models.
9. Evaluate each model with the metrics listed above and 5-fold stratified cross-validation.
10. Select the best model by macro F1, retrain on all available data, and save it as `heart_model.pkl`.
11. Use `predict_heart(file_path)` for single-file inference.

## Outputs

- `heart.ipynb` - main training and inference notebook
- `heart_model.pkl` - saved best model
- `label_encoder_heart.pkl` - saved label encoder

## Notes

- Classes with fewer than 5 samples are removed before training.
- Missing or unreadable files are skipped during feature extraction.
- The saved model and label encoder are required to reuse the trained pipeline outside the notebook.

## Future Work

- Mel-spectrogram based classification
- CNN-based heart sound recognition
- Source separation for mixed audio inputs
- Real-time auscultation analysis

## Citation

If you use this dataset or this notebook, please cite the HLS-CMDS dataset descriptor:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.