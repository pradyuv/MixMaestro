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
    
    # Extract labels
    labels = {
        'dynamic_range': np.array([feature['dynamic_range'] for feature in features]),
        # Add more labels as needed, just testing dynamic range right now
    }
    
    return features_array, labels

def train_model(features_array, labels):
    for label_name, label_values in labels.items():
        X_train, X_test, y_train, y_test = train_test_split(features_array, label_values, test_size=0.2, random_state=42)
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
        print(f"Mean Squared Error for {label_name}: {mse}")
        joblib.dump(model, os.path.abspath(f"models/{label_name}_model.pkl"))
        print(f"Model for {label_name} saved to models/{label_name}_model.pkl")

if __name__ == "__main__":
    features_array, labels = load_features("data/processed/features.pkl")
    train_model(features_array, labels)
