
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_digit_data(data_dir):
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.png'):
            # Parse label from filename (e.g., aug_digit_4_0.png)
            label = int(filename.split('_')[2])
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            img = img.astype(np.float32) / 255.0
            # Print shape for debugging
            print(f"Image {filename}: shape {img.shape}")
            images.append(img)
            labels.append(label)
    
    # Check shapes before converting to array
    shapes = [img.shape for img in images]
    unique_shapes = set(shapes)
    if len(unique_shapes) > 1:
        print(f"Inconsistent shapes detected: {unique_shapes}")
        for i, (img, shape) in enumerate(zip(images, shapes)):
            if shape != (28, 28):
                print(f"Image {i} has shape {shape}: {os.listdir(data_dir)[i]}")
    
    images = np.array(images)
    labels = np.array(labels)
    # Reshape images to (num_samples, 28, 28, 1)
    images = images.reshape(-1, 28, 28, 1)
    # Convert labels to categorical (0-9)
    labels = to_categorical(labels, num_classes=10)
    return images, labels

def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # 10 classes (0-9)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load data
    data_dir = "../data/augmented_digits"
    images, labels = load_digit_data(data_dir)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_cnn_model()
    model.summary()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Save the model
    model.save("../models/digit_cnn.h5")
    print("Model saved to ../models/digit_cnn.h5")
