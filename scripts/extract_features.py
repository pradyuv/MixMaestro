import librosa
import numpy as np
import os
import joblib
import pandas as pd

def extract_features(file_path):
    """
    Extracts audio features from a given file.

    Parameters: 
    file_path(str): The path to the audio file.

    Returns:
    dict: A dictionary containing extracted audio features.
    """
    y, sr = librosa.load(file_path)
    features = {
        'rms': librosa.feature.rms(y=y).mean(),  # measures power/loudness
        'peak_amplitude': np.max(np.abs(y)),  # measuring peak amplitude of signal
        'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),  # Represents perceived brightness
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=y).mean(),  # measures sign change of signal (detecting percussion)
        'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),  # measures width of spectrum (perceived timbre)
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),  # frequency below which a specified percentage of the total spectral energy lies, useful for distinguishing between harmonic/percussive sounds
        'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1),  # short term power spectrum (speech and audio processing)
        'chroma_stft': librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1),  # relates to 12 different pitch classes (notes), harmonic and chordal content analysis
        'tempo': librosa.beat.tempo(y=y, sr=sr)[0],  # detects tempo of audio
        'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1),  # measures difference in amplitude between peaks and valleys in the sound spectrum 
        'tonnetz': librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1),  # represents tonal properties, mapping to harmonic space
        'hnr': librosa.effects.harmonic(y=y).mean() / (librosa.effects.percussive(y=y).mean() + 1e-6),  # harmonic-to-noise ratio, distinguishing between harmonic content and noise
        'onset_strength': librosa.onset.onset_strength(y=y, sr=sr).mean(),  # measures strength of note onsets (rhythmic content)
        'pitch': librosa.core.piptrack(y=y, sr=sr)[0].max(),  # fundamental frequency of the signal
        'spectral_flatness': librosa.feature.spectral_flatness(y=y).mean(),  # indicates how noise-like the signal is
        'rolloff_frequency': librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85).mean(),  # frequency below which 85% of the spectral energy lies
        'perceptual_sharpness': librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1).std(),  # relates to perceived sharpness
        'perceptual_spread': librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1).mean(),  # relates to perceived spread
        'dynamic_range': np.max(librosa.feature.rms(y=y)) - np.min(librosa.feature.rms(y=y)),  # measures the difference between the quietest and loudest parts of the signal
    }
    return features

def process_directory(directory):
    """
    Processes all audio files in a given directory and extracts features from each file.

    Parameters: 
    directory (str): The path to the directory containing audio files.

    Returns:
    list: A list of dictionaries, each containing features of an audio file.
    """
    features = []

    for file in os.listdir(directory):
        if file.endswith(".wav"):
            file_path = os.path.join(directory, file)
            feature = extract_features(file_path)
            features.append(feature)

    return features

if __name__ == "__main__":
    raw_data_dir = os.path.abspath("data/raw/")
    processed_data_dir = os.path.abspath("data/processed/")
    
    print("Raw Data Directory:", raw_data_dir)
    print("Processed Data Directory:", processed_data_dir)

    if not os.path.exists(raw_data_dir):
        print(f"Error: The directory {raw_data_dir} does not exist.")
    else:
        features = process_directory(raw_data_dir)

        os.makedirs(processed_data_dir, exist_ok=True)

        joblib.dump(features, os.path.join(processed_data_dir, "features.pkl"))
        print(f"Features saved to {os.path.join(processed_data_dir, 'features.pkl')}")
