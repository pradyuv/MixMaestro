import joblib  # For loading the trained model
import numpy as np  # For numerical operations
import os  # For file path operations
from extract_features import extract_features # Importing functions from extract_features.py
from train_model import flatten_features  # Importing flatten_features from train_model.py

def predict(file_path, model_name):
    """ 
    Predicts the output for a new audio file using the specified trained model.

    Parameters:
    file_path (str): The path to the new audio file.
    model_name (str): The name of the model to use for prediction.

    Returns:
    numpy.ndarray: The model's prediction for the audio file.
    """
    model_path = os.path.abspath(model_name)
    model = joblib.load(model_path)

    # Extract features from the new audio file
    new_features = extract_features(file_path)
    # Flatten the features for prediction
    new_features_flat = flatten_features(new_features)
    # Convert the flattened features to a 2D numpy array
    new_features_array = np.array(new_features_flat).reshape(1, -1)
    # Make prediction using the loaded model
    prediction = model.predict(new_features_array)
    return prediction

if __name__ == "__main__":
    import argparse  # For handling command-line arguments

    # Setting up argument parser
    parser = argparse.ArgumentParser(description='Predict using trained model.')
    parser.add_argument('--file', type=str, required=True, help='Path to the new audio file.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model to use for prediction (e.g., models/mixing_model.pkl).')

    # Parsing the arguments
    args = parser.parse_args()
    new_audio_path = args.file
    model_name = args.model
    
    # Making prediction for the new audio file
    prediction = predict(new_audio_path, model_name)
    # Printing the prediction
    print(f"Prediction: {model_name}, {prediction}")
