# MixMaestro
The Mix Maestro is a tool aimed to help music producers and enthusiasts by automatically adjusting mixing parameters such as levels, equalization (EQ), and effects. My project attempts to leverage machine learning and real-time audio processing to provide intelligent mixing recommendations and adjustments. Passion project.

# Overview
MixMaestro extracts a comprehensive set of audio features from tracks to aid producers/enthusiasts (and potentially) mix engineers in making informed decisions while leaving room for personal mixing choices. The features extracted provide detailed insights into various aspects of the audio signal, such as volume, frequency content, pitch, rhythm, and more.

## Logic Flow

1. **User Interface (Java Web Application)**:
   - Users interact with the application through a web interface to upload their audio files and receive mixing recommendations.

2. **Backend Processing (Java, C++, Python)**:
   - **Java**: Manages file uploads and interacts with C++ and Python for processing.
   - **C++**: Handles advanced audio processing tasks, making it an integral part of the project.
   - **Python**: Handles feature extraction from processed audio, model training, and prediction using machine learning models.


### Explanation of C++ Processing

The C++ component of my program handles performance-critical and advanced audio processing tasks such as normalization, noise reduction, spectral analysis, and harmonic-percussive source separation. These tasks are computationally intensive and benefit from the efficiency and speed of C++ (versus Python). By preprocessing the audio in C++, the data passed to Python is **cleaner and more consistent**, enhancing the quality and reliability of the extracted features and leading to more accurate predictions and better mixing recommendations.

## Detailed Logic Flow

### 1. User Uploads Audio File

- **Step**: The user accesses the web application and uploads their unmixed audio file.
- **Technology**: Java (Spring Boot), React.js (possibly)

### 2. Backend Receives the File

- **Step**: The Java backend receives the uploaded file and saves it to a designated location.
- **Technology**: Java (Spring Boot)

### 3. Invoke C++ for Advanced Processing

- **Step**: The Java backend calls C++ functions (using JNI) to perform advanced audio processing tasks.
- **Technology**: C++
- **Tasks in C++**:
  - **Normalization**: Adjusts the audio signal to a standard level.
  - **Noise Reduction**: Reduces background noise.
  - **Spectral Analysis**: Analyzes the frequency spectrum using [FFT](https://www.nti-audio.com/en/support/know-how/fast-fourier-transform-fft#:~:text=The%20%22Fast%20Fourier%20Transform%22%20(,frequency%20information%20about%20the%20signal.).
  - **Harmonic-Percussive Source Separation**: Separates harmonic and percussive components.

### 4. Pass Processed Audio to Python for Feature Extraction

- **Step**: The processed audio data from C++ is passed to Python scripts for detailed feature extraction.
- **Technology**: Python

### 5. Python Feature Extraction and Prediction

- **Sub-Steps**:
  - **Feature Extraction (`extract_features.py`)**: Extract detailed features from the processed audio data.
  - **Prediction (`predict.py`)**: Use the trained model to analyze the features and provide mixing recommendations.
- **Technology**: Python

### 6. Return Recommendations to User

- **Step**: The Python script returns the analysis results to the Java backend.
- **Technology**: Python, Java (Spring Boot)
- **Final Step**: The Java backend sends the recommendations back to the user via the web interface.
- **Technology**: Java (Spring Boot)

## Components

### 1. C++ Audio Processing

Handles advanced audio processing tasks.
- **Files**: `audio_processor.h`, `audio_processor.cpp`, `main.cpp`
- **Build Configuration**: `CMakeLists.txt`

### 2. JNI Integration

Allows Java to call C++ functions.
- **Java Class**: `NativeAudioProcessor.java`
- **Generated C++ Header**: `NativeAudioProcessor.h`
- **JNI Implementation**: `NativeAudioProcessor.cpp`

### 3. Java Web Application (Spring Boot)

Manages the user interface and backend processing.
- **Main Application**: `MixingAssistantApplication.java`
- **REST Controller**: `AudioController.java`
- **Build Configuration**: `build.gradle`

### 4. Python Scripts

Handles feature extraction and machine learning.
- **Feature Extraction**: `extract_features.py`
- **Model Training**: `train_model.py`
- **Prediction**: `predict.py`


# extract_features.py

  ## Purpose

  The extract_features.py script is designed to extract a wide range of audio features from an audio file. These features provide essential information about the audio signal's characteristics, which can be used for intelligent/learned mixing and audio analysis.  

  ## How It Works

  The script utilizes the [librosa library] (https://librosa.org/doc/latest/index.html) to load audio files and extract various features. It processes multiple audio files in a directory, extracts the features, and saves them for further analysis or use in machine learning models.

  ## Key Components

  1. **Import Libraries**:
      ```
      python
      import librosa  # Import the librosa library for audio processing
      import numpy as np  # Import NumPy for numerical operations
      import os  # Import the os library for file and directory operations
      import joblib  # Import the joblib library for saving and loading data
      ```

  2. **Feature Extraction Function**:

    Can refer to scripts/extract_features.py for code.

  **Detailed Feature Explanation**:

- **RMS Energy (rms):** Measures the power or loudness of the track.
- **Peak Amplitude (peak_amplitude):** Measures the peak amplitude of the signal, useful for detecting clipping.
- **Spectral Centroid (spectral_centroid):** Indicates the brightness of the track.
- **Zero-Crossing Rate (zero_crossing_rate):** Measures the rate at which the signal changes sign, useful for detecting percussive elements.
- **Spectral Bandwidth (spectral_bandwidth):** Provides information about the spread of frequencies, related to the perceived timbre of the sound.
- **Spectral Roll-off (spectral_rolloff):** The frequency below which a specified percentage of the total spectral energy lies, useful for distinguishing between harmonic and percussive sounds.
- **MFCCs (mfcc):** Represents the short-term power spectrum of a sound, widely used in speech and audio processing.
- **Chroma Features (chroma_stft):** Reflects the energy distribution across the 12 different pitch classes, useful for harmonic and chordal content analysis.
- **Tempo (tempo):** The tempo of the audio signal, expressed in beats per minute (BPM), essential for synchronizing tracks.
- **Spectral Contrast (spectral_contrast):** Measures the difference in amplitude between peaks and valleys in the sound spectrum, providing insights into the harmonic content and dynamics of the audio signal.
- **Tonnetz (tonnetz):** Maps the audio into a harmonic space, representing the tonal properties of the sound.
- **Harmonic-to-Noise Ratio (hnr):** Measures the ratio of harmonic sound to noise, useful for distinguishing between harmonic content and noise.
- **Onset Strength (onset_strength):** Measures the strength of note onsets in the audio signal, useful for identifying and emphasizing transient events such as drum hits or note attacks.
- **Pitch (pitch):** The fundamental frequency of the audio signal, useful for pitch correction, harmony generation, and aligning tracks with different pitches.
- **Spectral Flatness (spectral_flatness):** Measures how flat the spectrum is, indicating whether a signal is more noise-like or tone-like.
- **Rolloff Frequency (rolloff_frequency):** The frequency below which 85% of the spectral energy lies, useful for adjusting filters and ensuring a balanced frequency response.
- **Perceptual Sharpness (perceptual_sharpness):** Relates to the perceived sharpness of the sound.
- **Perceptual Spread (perceptual_spread):** Relates to the perceived spread of the sound.
- **Dynamic Range (dynamic_range):** Measures the difference between the quietest and loudest parts of the signal, useful for managing compression and expansion to maintain the desired dynamic feel.

    

3. **Directory Processing Function**:

  The process_directory function processes all .wav files in a given directory, extracting features from each file using the extract_features function. It returns a list of dictionaries, each containing the features of an audio file.

4. **Main Block for Script Execution**:
 
  The main block defines the paths to the raw and processed data directories, processes the directory of raw audio files to extract features, and saves the extracted features to a file using joblib.


        

# train_model.py

## Purpose

The `train_model.py` script is designed to train a machine learning model using the features extracted from audio files (`extract_features.py`). The trained model can then be used to make intelligent predictions or adjustments for mixing tasks.

## How It Works

The script loads the extracted features, processes them to ensure they are in a suitable format, splits the data into training and testing sets, trains a `RandomForestRegressor` model, evaluates the model, and saves the trained model for future use.

### Key Components

1. **Importing Libraries**:
    - The script starts by importing necessary libraries such as `joblib` for loading and saving models, `numpy` for numerical operations, and `sklearn` for machine learning algorithms and utilities.
    ```python
    import joblib
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    ```

2. **Loading Extracted Features**:
    - Loads the features extracted by `extract_features.py` using `joblib`.
    ```python
    features = joblib.load(os.path.abspath("data/processed/features.pkl"))
    ```

3. **Flattening the Feature Dictionaries**:
    - The extracted features are initially stored as dictionaries with various data types (scalars, lists, arrays). To make these features suitable for model training, they need to be converted into a flat list of numeric values.
    - The `flatten_features` function processes each feature dictionary, flattening arrays and extending lists/tuples to ensure all values are included in a single, flat list.
    ```python
    def flatten_features(feature_dict):
        flat_features = []
        for key, value in feature_dict.items():
            if isinstance(value, np.ndarray):
                flat_features.extend(value.flatten())
            elif isinstance(value, (list, tuple)):
                flat_features.extend(value)
            else:
                flat_features.append(value)
        return flat_features
    ```

4. **Converting to a NumPy Array**:
    - After flattening the features, they are converted to a 2D NumPy array where each row represents the features for one audio file.
    - Ensuring the array has a `float` data type guarantees that all elements are numeric, which is necessary for our model training.
    ```python
    features_array = np.array([flatten_features(feature) for feature in features])
    features_array = np.array(features_array, dtype=float)
    ```

5. **Creating Dummy Labels**:
    - For initial testing purposes, dummy labels are generated using random numbers. These labels will be replaced with meaningful labels when training the model for practical use in a later implementation once I have developed the application further.

### Importance of Replacing Dummy Labels with Actual Targets

To train a machine learning model effectively, the labels (or targets) must represent the specific outcomes or values that the model is being trained to predict (EQ, dynamic range etc.). Dummy labels are placeholders used for initial testing. Replacing these with actual targets is crucial for the model to learn meaningful patterns and make accurate predictions.

### Steps to Replace Dummy Labels with Actual Targets

- **Define the Prediction Task**:
    - Determine what we want the model to predict. For example, in a mixing context, you might want to predict optimal EQ settings, volume levels, or other audio mixing parameters.

- **Collect Target Data**:
    - Gather data for the actual targets you want to predict. This might involve manually annotating the audio files with the desired outcomes, extracting target values from existing mixed tracks, or using some form of automated measurement.

- **Format the Target Data**:
    - Ensure that the target data is in a format compatible with your model. Typically, this means having a list or array of numeric values corresponding to each audio file's features.

- **Integrate the Target Data into the Script**:
    - Modify the `train_model.py` script to use the actual targets instead of the dummy labels.
    ```python
    labels = np.random.rand(len(features_array)) <- on this line, I will eventually edit to fit real-world parameters
    ```

6. **Splitting Data**:
    - The data is split into training and testing sets to evaluate the model's performance.
    ```python
    X_train, X_test, y_train, y_test = train_test_split(features_array, labels, test_size=0.2, random_state=42)
    ```

7. **Initializing and Training the Model**:
    - A `RandomForestRegressor` is initialized with specified hyperparameters and trained on the training data.
    ```python
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    ```

8. **Evaluating the Model**:
    - The model is evaluated on the testing set using Mean Squared Error (MSE) to measure its performance.
    ```python
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    ```

9. **Saving the Trained Model**:
    - The trained model is saved to the `models/` directory for future use.
    ```python
    model_path = os.path.abspath("models/mixing_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    ```

### Importance of Splitting Data into Training and Testing Sets

Splitting data into training and testing sets is a critical step in the machine learning workflow for several reasons:

1. **Model Evaluation**:
   - **Generalization**: This is more to see how well the model we train can generalize to new and unseen data. When we train the model on one subset of the data then test it on another, we can assess how well the model is likely to perform on real-world data (your own mixes, or a large repertoire of professionally mixed/mastered songs).
   - **Overfitting Detection**: If our model performs well on the training data but poorly on the testing set, this is defined as overfitting. This means that the model has learned the training data **too well**, including its [noise](https://www.iguazio.com/glossary/noise-in-ml/).

2. **Performance Metrics**:
   - **Unbiased Estimate**: Provides an unbiased estimate of the model's performance, with metrics calculated on the testing set reflecting the model's ability to handle new data.

3. **Model Validation**:
   - **Hyperparameter Tuning**: Allows for fine-tuning of hyperparameters (config settings used to control behavior/performance of ML algos) based on performance on the testing set.
   - **Model Selection**: Helps in comparing different models or algorithms and selecting the best one.

4. **Preventing Data Leakage**:
   - **Data Leakage Avoidance**: Ensures that information from the testing set does not influence the training process, which could lead to misleading performance metrics.



