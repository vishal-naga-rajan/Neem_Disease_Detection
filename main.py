import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

DATASET_DIR = 'neem_leaf_dataset'

TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = len(os.listdir(TRAIN_DIR))

print("Setting up data generators...")
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Use MobileNetV2's preprocess_input
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation data, we only apply the preprocessing function.
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create data generators that will read images from the directories
print(f"Looking for training images in: {TRAIN_DIR}")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print(f"Looking for validation images in: {VAL_DIR}")
validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
print("Building the model with MobileNetV2 base...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer with dropout for regularization.
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
# Add the final output layer with softmax activation for our classes.
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("Compiling the model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Start with a standard learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print a summary of the model architecture
model.summary()


# --- Callbacks ---
# Define callbacks to improve the training process.
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5, # Reduce patience as transfer learning converges faster
    verbose=1,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'neem_disease_detector_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Reduce learning rate when a metric has stopped improving.
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.00001,
    verbose=1
)


# --- Train the Model ---
print("Starting model training...")
EPOCHS = 50 # We might not need as many epochs with transfer learning
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr] # Add all callbacks
)

# --- Evaluate the Model & Visualize Results ---
print("Training finished. Evaluating model...")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# --- Save the Final Model ---
# The best model is already saved by ModelCheckpoint as 'neem_disease_detector_best.h5'.
# We can also save the final trained model state.
model.save('neem_disease_detector_final.h5')
print("Final model saved as 'neem_disease_detector_final.h5'")
print("Best model saved as 'neem_disease_detector_best.h5'")

# To get the class indices which will be useful for prediction
class_indices = train_generator.class_indices
print("\nClass Indices:", class_indices)
# Save class indices to a file for later use in the Flask app
import json
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
print("Class indices saved to 'class_indices.json'")
