# Mix Sound Classification

This notebook evaluates heart and lung classification models on mixed audio recordings (combined heart + lung sounds). It applies the same feature extraction used for the single-task notebooks and runs both the heart and lung models on each mixed recording to measure how well each model performs when recordings contain mixed sources.

Dataset
-------

The mixed recordings used for evaluation are part of the UCI HLS-CMDS collection:

https://archive.ics.uci.edu/dataset/1202/hls-cmds:+heart+and+lung+sounds+dataset+recorded+from+a+clinical+manikin+using+digital+stethoscope

Please cite the dataset descriptor:

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.

Notebook workflow
-----------------

`mix.ipynb` follows these steps:

1. Load saved models: `heart_model.pkl` and `lung_model.pkl` (the heart and lung notebooks include examples of saving trained models using `joblib`).
2. Load `dataset/Mix.csv` and iterate over `Mixed Sound ID` values to locate `dataset/Mix/*.wav` files.
3. For each mixed `.wav` file, extract the same compact feature vector used in the single-task notebooks (20 MFCC means + zcr + rms + spectral centroid + bandwidth).
4. Run both the heart and lung models on the same feature vector and record predictions alongside the true labels from `Mix.csv`.
5. Aggregate results into `mix_evaluation_results.csv` and compute per-task accuracy, classification reports, and confusion matrices.

Files produced
--------------

- `mix_evaluation_results.csv` — per-file true vs predicted labels for heart and lung tasks
- Confusion matrix plots — visualized inside the notebook

Usage
-----

1. Ensure `heart_model.pkl` and `lung_model.pkl` exist in their respective folders (or update `mix.ipynb` to point to alternate paths).
2. Confirm `dataset/Mix.csv` and WAV files are present in `dataset/Mix/`.
3. Run `mix.ipynb` from top to bottom. The notebook will save `mix_evaluation_results.csv` in the working directory.

Limitations and notes
---------------------

- The approach uses features averaged across the entire recording; mixed signals may require representation methods that preserve temporal or source-separation information for best performance.
- If heart and lung sounds overlap heavily in frequency/time, classical per-file features may be insufficient — consider source separation or deep models that operate on spectrograms.

Future enhancement: CNNs and spectrogram-based models
----------------------------------------------------

As a next step, replace the aggregated features with spectrograms or mel-spectrogram images and train convolutional neural networks (CNNs). CNNs can learn spatial patterns in time-frequency representations and may better separate and classify overlapping sources in mixed recordings.

Citation
--------

Y. Torabi, S. Shirani and J. P. Reilly, "Descriptor: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope (HLS-CMDS)," in IEEE Data Descriptions, doi: 10.1109/IEEEDATA.2025.3566012.
