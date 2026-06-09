# Heart & Lung Sound Classification Using Machine Learning (HLS-CMDS)

A machine learning framework for heart sound classification, lung sound classification, and mixed auscultation recording evaluation using the HLS-CMDS dataset.

## Overview

Auscultation remains one of the most widely used non-invasive diagnostic techniques for identifying cardiovascular and respiratory disorders. Automated analysis of heart and lung sounds has the potential to assist healthcare professionals by providing objective and scalable diagnostic support.

This project investigates the classification of heart sounds, lung sounds, and mixed auscultation recordings using traditional machine learning techniques. Separate classifiers are trained on isolated heart and lung recordings and subsequently evaluated on mixed recordings to study their robustness under realistic clinical conditions.

## Research Objectives

* Develop machine learning models for heart sound classification.
* Develop machine learning models for lung sound classification.
* Compare the performance of Random Forest, Support Vector Machine (SVM), and XGBoost classifiers.
* Evaluate model robustness on mixed heart-lung recordings.
* Analyze performance degradation and class confusion in mixed audio environments.

## Dataset

This project uses the HLS-CMDS (Heart and Lung Sounds Dataset) dataset recorded from a clinical manikin using a digital stethoscope.

Dataset source:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

### Dataset Structure

```text
dataset/
├── HS.csv
├── LS.csv
├── Mix.csv
├── HS/
│   └── *.wav
├── LS/
│   └── *.wav
└── Mix/
    └── *.wav
```

Components:

* HS.csv and HS/ contain heart sound metadata and recordings.
* LS.csv and LS/ contain lung sound metadata and recordings.
* Mix.csv and Mix/ contain mixed heart-lung recordings.

Each metadata record references a corresponding WAV audio file.

## Repository Structure

```text
.
├── heart_sound_classification/
│   └── heart.ipynb
│
├── lung_sound_classification/
│   └── lung.ipynb
│
├── mix_sound_classification/
│   └── mix.ipynb
│
├── dataset/
│
└── requirements.txt
```

## Methodology

### Feature Extraction

Each audio recording is loaded at 22,050 Hz and converted into a fixed-length feature vector using Librosa.

Extracted features:

* MFCC means (20)
* MFCC standard deviations (20)
* Chroma features (12)
* Spectral contrast (7)
* Zero Crossing Rate (ZCR)
* RMS Energy
* Spectral Centroid
* Spectral Bandwidth
* Spectral Rolloff

These features form a compact representation suitable for traditional machine learning algorithms.

### Machine Learning Models

The following classifiers are evaluated:

* Dummy Classifier (Baseline)
* Random Forest
* Support Vector Machine (SVM)
* XGBoost

All models are trained using:

* StandardScaler preprocessing
* Stratified train-test split
* 5-fold Stratified Cross Validation
* Macro F1-based model comparison

### Evaluation Metrics

Performance is evaluated using:

* Accuracy
* Macro F1 Score
* Classification Report
* Confusion Matrix
* Stratified K-Fold Cross Validation

## Experimental Workflow

### Heart Sound Classification

The heart classification notebook:

* Loads heart sound recordings.
* Extracts acoustic features.
* Trains and evaluates multiple classifiers.
* Selects the best-performing model.
* Saves the trained model and label encoder.

Outputs:

```text
heart_model.pkl
label_encoder_heart.pkl
```

### Lung Sound Classification

The lung classification notebook:

* Loads lung sound recordings.
* Uses the same feature extraction pipeline.
* Trains and evaluates identical model architectures.
* Saves the best-performing model.

Outputs:

```text
lung_model.pkl
label_encoder_lung.pkl
```

### Mixed Recording Evaluation

The mixed recording notebook:

* Loads the trained heart and lung models.
* Extracts features from mixed recordings.
* Predicts heart and lung labels independently from the same audio signal.
* Evaluates classifier robustness under mixed-source conditions.

Outputs:

```text
mix_evaluation_results.csv
```

## Running the Project

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Execute Notebooks

Run the notebooks in the following order:

1. heart_sound_classification/heart.ipynb
2. lung_sound_classification/lung.ipynb
3. mix_sound_classification/mix.ipynb

## Expected Outputs

```text
heart_sound_classification/
├── heart_model.pkl
└── label_encoder_heart.pkl

lung_sound_classification/
├── lung_model.pkl
└── label_encoder_lung.pkl

mix_sound_classification/
└── mix_evaluation_results.csv
```

## Key Research Contribution

Unlike conventional heart-only or lung-only classification studies, this project investigates how classifiers trained on isolated biomedical signals perform when exposed to mixed auscultation recordings.

The resulting analysis provides insight into:

* Model robustness under signal interference.
* Performance degradation in mixed environments.
* Practical challenges of real-world automated auscultation systems.

## Limitations

* Global statistical features do not preserve temporal information.
* Mixed recordings contain overlapping physiological signals that increase classification difficulty.
* Traditional machine learning models may struggle to separate concurrent sound sources.

## Future Work

Potential extensions include:

* Convolutional Neural Networks (CNNs)
* Mel-spectrogram-based classification
* Audio source separation techniques
* Real-time auscultation analysis
* Multi-task learning for simultaneous heart and lung diagnosis
* Transformer-based biomedical audio models

## Citation

If you use this dataset, please cite:


Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.