# main_hybrid_train.py
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
import os
import joblib # For saving sklearn models

# Import scikit-learn classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- Configuration ---
DATASET_DIR = 'neem_leaf_dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# --- Step 1: Build the Feature Extractor Model ---
print("Step 1: Building the feature extractor model...")

# Load the base MobileNetV2 model, excluding the final classification layer
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False # Freeze the base model

# Create a new model that outputs the feature vectors
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

print("Feature extractor built successfully.")
feature_extractor.summary()


# --- Step 2: Extract Features from the Dataset ---
print("\nStep 2: Extracting features from the dataset...")

# Create a data generator (no augmentation needed for feature extraction)
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Function to extract features from a directory
def extract_features(directory):
    print(f"Extracting features from {directory}...")
    generator = datagen.flow_from_directory(
        directory,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # IMPORTANT: Do not shuffle to keep labels in order
    )
    # Use the model to predict (extract features)
    features = feature_extractor.predict(generator, steps=len(generator), verbose=1)
    labels = generator.classes
    return features, labels

# Extract features for both training and validation sets
X_train, y_train = extract_features(TRAIN_DIR)
X_val, y_val = extract_features(VAL_DIR)

print(f"Training features shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation features shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")


# --- Step 3: Train and Evaluate Machine Learning Classifiers ---
print("\nStep 3: Training and evaluating traditional ML classifiers...")

# Initialize the models
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

best_classifier = None
best_accuracy = 0.0

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"\n--- Training {name} ---")
    
    # Train the model (this will be very fast)
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Validation Accuracy for {name}: {accuracy * 100:.2f}%")

    # Keep track of the best performing model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = clf
        best_classifier_name = name

print(f"\nBest performing model is {best_classifier_name} with {best_accuracy * 100:.2f}% accuracy.")


# --- Step 4: Save the Models and Class Indices ---
print("\nStep 4: Saving the necessary components...")

# 1. Save the best performing ML classifier
if best_classifier is not None:
    joblib.dump(best_classifier, f'{best_classifier_name.replace(" ", "_").lower()}_classifier.joblib')
    print(f"Best classifier ({best_classifier_name}) saved as '{best_classifier_name.replace(' ', '_').lower()}_classifier.joblib'")

# 2. Save the Keras feature extractor model
feature_extractor.save('feature_extractor.h5')
print("Keras feature extractor saved as 'feature_extractor.h5'")

# 3. Save the class indices for mapping predictions to class names
# We need to create a generator one more time to get the class_indices map
train_generator_for_indices = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
class_indices = train_generator_for_indices.class_indices
print("\nClass Indices:", class_indices)

import json
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
print("Class indices saved to 'class_indices.json'")

print("\nProcess completed successfully! ✨")