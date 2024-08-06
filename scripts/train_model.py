import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("Current Working Directory:", os.getcwd())

# Load extracted features (extract_features.py)
features = joblib.load(os.path.abspath("data/processed/features.pkl"))

# Flatten the feature dictionaries
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

# Convert list of feature dictionaries to a 2D numpy array (dict to flattened array)
features_array = np.array([flatten_features(feature) for feature in features])

# Ensure all elements in features_array are numeric
features_array = np.array(features_array, dtype=float)

# Create dummy labels for testing (replace with actual labels in practice, in a later implementation)
labels = np.random.rand(len(features_array))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_array, labels, test_size=0.2, random_state=42)

# Initialize and train the model with specified hyperparameters
model = RandomForestRegressor(
    n_estimators=200,          # Number of trees in the forest
    max_depth=10,              # Maximum depth of the trees
    min_samples_split=5,       # Minimum number of samples required to split an internal node
    min_samples_leaf=4,        # Minimum number of samples required to be at a leaf node
    max_features='sqrt',       # Number of features to consider when looking for the best split
    bootstrap=True,            # Whether bootstrap samples are used when building trees
    random_state=42,           # Ensures reproducibility
    n_jobs=-1                  # Use all processors for parallel processing
)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save the trained model
model_path = os.path.abspath("models/mixing_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
