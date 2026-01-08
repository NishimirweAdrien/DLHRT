import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_mnist_data():
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Combine train and test sets for a unified split
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    # Normalize pixel values to [0, 1]
    X = X.astype(np.float32) / 255.0
    
    # Reshape images to (num_samples, 28, 28, 1)
    X = X.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical (0-9)
    y = to_categorical(y, num_classes=10)
    
    return X, y

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
    # Load MNIST data
    print("Loading MNIST data...")
    images, labels = load_mnist_data()
    print(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}")

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build and train the model
    print("Building and training model...")
    model = build_cnn_model()
    model.summary()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Save the model
    model_path = "../models/digit_cnn_mnist.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")