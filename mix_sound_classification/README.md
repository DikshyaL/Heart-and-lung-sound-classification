# Mix Sound Classification

This folder contains the mixed-recording evaluation workflow implemented in [mix.ipynb](mix.ipynb). Unlike the heart and lung notebooks, this notebook does not train new models. It loads the saved heart and lung models, applies them to mixed recordings, and measures how the single-task models behave when the audio contains overlapping sources.

## Dataset

The mixed recordings come from the UCI HLS-CMDS collection:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

Inputs used by the notebook:

- `dataset/Mix.csv` for labels and metadata
- `dataset/Mix/` for the corresponding `.wav` files
- `heart_sound_classification/heart_model.pkl`
- `lung_sound_classification/lung_model.pkl`
- `heart_sound_classification/label_encoder_heart.pkl`
- `lung_sound_classification/label_encoder_lung.pkl`

## Workflow

The notebook follows an evaluation-only pipeline:

1. Load the persisted heart and lung models with `joblib.load`.
2. Load the corresponding saved label encoders.
3. Read `Mix.csv` and iterate through each `Mixed Sound ID`.
4. Build the path to the matching file in `dataset/Mix/`.
5. Extract the same 64-dimensional feature vector used in the single-task notebooks.
6. Predict both heart and lung labels from the same feature vector.
7. Compare the predictions with the true labels in `Mix.csv`.
8. Save the per-record results to `mix_evaluation_results.csv`.
9. Report accuracy, classification metrics, and confusion matrices for both tasks.

## Architecture

The notebook is built as a lightweight inference and analysis layer on top of the trained heart and lung models.

### Shared feature extractor

The mixed recordings are converted into the same 64-dimensional summary feature vector used in the other notebooks:

- 20 MFCC means
- 20 MFCC standard deviations
- 12 chroma features
- 7 spectral contrast values
- zero-crossing rate
- RMS energy
- spectral centroid
- spectral bandwidth
- spectral rolloff

This keeps the evaluation consistent across all three tasks.

### Prediction layer

The notebook uses the already-trained classifiers from the single-task notebooks:

- heart model loaded from `heart_model.pkl`
- lung model loaded from `lung_model.pkl`

Each model receives the same input vector, but the notebook evaluates them against different ground-truth labels from `Mix.csv`.

### Evaluation layer

The notebook computes:

- accuracy
- classification report
- confusion matrix
- macro F1 score, where applicable in the notebook output

## Outputs

- `mix_evaluation_results.csv` - per-file true labels and predictions for the heart and lung tasks
- confusion matrix plots - rendered inside the notebook

## Usage

1. Run [heart.ipynb](../heart_sound_classification/heart.ipynb) and [lung.ipynb](../lung_sound_classification/lung.ipynb) first so the model and encoder files exist.
2. Confirm that `dataset/Mix.csv` and the WAV files in `dataset/Mix/` are available.
3. Open [mix.ipynb](mix.ipynb) and run the cells from top to bottom.

If the saved artifacts are in a different location, update the paths in the notebook before running it.

## Notes

- This notebook is an evaluation harness, not a training notebook.
- Mixed audio can be harder to classify because the feature representation is still a single global summary of the full recording.
- If heart and lung sounds overlap strongly, spectrogram-based or source-separation approaches will likely perform better.

## Citation

If you use this dataset or this notebook, please cite the HLS-CMDS dataset descriptor:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.
