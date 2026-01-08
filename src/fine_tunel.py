import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Load the pre-trained model
model = load_model("digit_recognition_cnn.h5")

# Unfreeze more layers for better adaptation
for layer in model.layers[:-4]:  # Freeze fewer layers
    layer.trainable = False
for layer in model.layers[-4:]:  # Unfreeze the last 4 layers
    layer.trainable = True

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])

# Enhanced data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Train generator
train_generator = datagen.flow_from_directory(
    'dataset/',
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

# Validation generator
validation_generator = datagen.flow_from_directory(
    'dataset/',
    target_size=(28, 28),
    color_mode='grayscale',
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Fine-tune with more epochs
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // validation_generator.batch_size),
    epochs=10
)

# Save the fine-tuned model
model.save("digit_recognition_cnn_finetuned.h5")