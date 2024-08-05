# MixMaestro
The Mix Maestro is a tool aimed to help music producers and enthusiasts by automatically adjusting mixing parameters such as levels, equalization (EQ), and effects. My project attempts to leverage machine learning and real-time audio processing to provide intelligent mixing recommendations and adjustments. Passion project.

# Overview
MixMaestro extracts a comprehensive set of audio features from tracks to aid producers/enthusiasts (and potentially) mix engineers in making informed decisions while leaving room for personal mixing choices. The features extracted provide detailed insights into various aspects of the audio signal, such as volume, frequency content, pitch, rhythm, and more.



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


        

    

