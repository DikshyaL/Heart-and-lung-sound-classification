# Mix Sound Classification

This folder contains the mixed-recording evaluation workflow implemented in [mix.ipynb](mix.ipynb). Unlike the heart and lung notebooks, this notebook does not train new models. It loads the saved heart and lung models, applies them to mixed recordings, and measures how the single-task classifiers behave when the audio contains overlapping sources.

## Project Summary

The mix notebook evaluates how well models trained on isolated heart and lung recordings generalize to mixed auscultation recordings. It is an evaluation-only layer designed to expose robustness limits in realistic overlapping-signal conditions.

## Dataset

The mixed recordings come from the HLS-CMDS collection:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

Inputs used by the notebook:

- `dataset/Mix.csv` for labels and metadata
- `dataset/Mix/` for the corresponding WAV files
- `heart_sound_classification/heart_model.pkl`
- `lung_sound_classification/lung_model.pkl`
- `heart_sound_classification/label_encoder_heart.pkl`
- `lung_sound_classification/label_encoder_lung.pkl`

## Feature Extraction

The mixed recordings are converted into the same 64-dimensional summary feature vector used in the heart and lung notebooks:

- 20 MFCC means
- 20 MFCC standard deviations
- 12 chroma features
- 7 spectral contrast values
- zero-crossing rate
- RMS energy
- spectral centroid
- spectral bandwidth
- spectral rolloff

Keeping the representation identical across all tasks makes the mixed-recording evaluation directly comparable to the single-task experiments.

## Evaluation Metrics

The notebook computes:

- Accuracy
- Macro F1 score
- Classification report
- Confusion matrix

## Workflow

1. Load the persisted heart and lung models with `joblib.load`.
2. Load the corresponding saved label encoders.
3. Read `Mix.csv` and iterate through each `Mixed Sound ID`.
4. Build the path to the matching file in `dataset/Mix/`.
5. Extract the same 64-dimensional feature vector used in the single-task notebooks.
6. Predict both heart and lung labels from the same feature vector.
7. Compare the predictions with the true labels in `Mix.csv`.
8. Save the per-record results to `mix_evaluation_results.csv`.
9. Report accuracy, classification metrics, and confusion matrices for both tasks.

## Outputs

- `mix.ipynb` - main evaluation notebook
- `mix_evaluation_results.csv` - per-file true labels and predictions for the heart and lung tasks
- confusion matrix plots - rendered inside the notebook

## Notes

- This notebook is an evaluation harness, not a training notebook.
- Mixed audio is harder to classify because the feature representation is still a single global summary of the full recording.
- If heart and lung sounds overlap strongly, spectrogram-based or source-separation approaches will likely perform better.

## Future Work

- Source separation before classification
- Mel-spectrogram-based evaluation
- Real-time mixed auscultation analysis

## Citation

If you use this dataset or this notebook, please cite the HLS-CMDS dataset descriptor:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.
