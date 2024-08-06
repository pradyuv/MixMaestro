import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

def load_features(filepath):
    features = joblib.load(filepath)
    features_array = np.array([flatten_features(feature) for feature in features])
    features_array = np.array(features_array, dtype=float)
    return features_array

def train_model(features_array, labels):
    X_train, X_test, y_train, y_test = train_test_split(features_array, labels, test_size=0.2, random_state=42)
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
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return model

if __name__ == "__main__":
    features_array = load_features("data/processed/features.pkl")
    labels = np.random.rand(len(features_array))  # Replace with actual labels
    model = train_model(features_array, labels)
    model_path = os.path.abspath("models/mixing_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
