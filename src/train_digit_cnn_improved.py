
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load and preprocess digit images
def load_digit_images(data_dir="../data/augmented_digits"):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.startswith("aug_digit_") and filename.endswith(".png"):
            # Extract label from filename (e.g., aug_digit_5_*.png -> label 5)
            label = int(filename.split("_")[2])
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            # Ensure the image is 28x28
            img = cv2.resize(img, (28, 28))
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load the dataset
print("Loading digit images...")
images, labels = load_digit_images()
if len(images) == 0:
    raise ValueError("No images found in the dataset directory.")

# Reshape images for CNN input
images = images.reshape(-1, 28, 28, 1)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.3, random_state=42  # Increased validation split
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Define early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model with data augmentation
print("Training the model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=50,  # Increased epochs
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Save the model
model.save("../models/digit_cnn.h5")
print("Model saved to ../models/digit_cnn.h5")

# Print final accuracies
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final training accuracy: {final_train_acc:.4f}")
print(f"Final validation accuracy: {final_val_acc:.4f}")
