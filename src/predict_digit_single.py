import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import os

# 1. Load the saved model
try:
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'mnist_cnn_model.h5' exists in the same directory.")
    exit(1)

# 2. Function to preprocess and predict digit from a custom image
def predict_digit(image_path, invert_colors=True, enhance_contrast=True):
    try:
        # Load the image with PIL
        image = Image.open(image_path)
        print(f"Image format: {image.format}")
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")

        # Convert to grayscale
        img = image.convert("L")
        img.save('debug_grayscale.png')
        print("Grayscale image saved as 'debug_grayscale.png'")

        # Enhance contrast if specified
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)  # Adjustable contrast
            img.save('debug_contrast.png')
            print("Contrast-enhanced image saved as 'debug_contrast.png'")

        # Apply sharpening filter
        img = img.filter(ImageFilter.SHARPEN)

        # Convert to numpy array for OpenCV processing
        img_array = np.array(img)

        # Print grayscale statistics
        print(f"Grayscale stats - Min: {img_array.min()}, Max: {img_array.max()}, Mean: {img_array.mean():.2f}")

        # Apply Otsu's thresholding to binarize the image
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite('debug_threshold.png', img_array)
        print("Thresholded image saved as 'debug_threshold.png'")

        # Apply morphological operations to denoise and thicken the digit
        kernel = np.ones((3, 3), np.uint8)
        img_array = cv2.dilate(img_array, kernel, iterations=1)  # Thicken the digit
        img_array = cv2.erode(img_array, kernel, iterations=1)  # Remove small noise
        cv2.imwrite('debug_morphology.png', img_array)
        print("Image after morphological operations saved as 'debug_morphology.png'")

        # Find contours to center the digit
        contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Filter contours by minimum area to avoid noise
            contours = [c for c in contours if cv2.contourArea(c) > 50]  # Adjust 50 as needed
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Crop the digit with some padding
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img_array.shape[1] - x, w + 2 * padding)
                h = min(img_array.shape[0] - y, h + 2 * padding)
                digit_crop = img_array[y:y+h, x:x+w]
                cv2.imwrite('debug_cropped.png', digit_crop)
                print("Cropped digit saved as 'debug_cropped.png'")

                # Resize to 20x20 (to fit within 28x28 with border)
                digit_resized = cv2.resize(digit_crop, (20, 20), interpolation=cv2.INTER_AREA)

                # Create a 28x28 canvas and center the digit
                canvas = np.zeros((28, 28), dtype=np.uint8)
                x_offset = (28 - 20) // 2
                y_offset = (28 - 20) // 2
                canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digit_resized
                img_array = canvas
            else:
                print("Warning: No valid contours found after filtering. Using original image for resizing.")
                img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            print("Warning: No contours found. Using original image for resizing.")
            img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)

        # Invert colors if specified (MNIST: white digits on black background)
        if invert_colors:
            img_array = 255 - img_array

        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0

        # Save final preprocessed image
        cv2.imwrite('preprocessed_image.png', img_array * 255)
        print("Final preprocessed image saved as 'preprocessed_image.png'")

        # Reshape for model input
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict the digit
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction[0])
        probabilities = prediction[0]

        # Print prediction probabilities
        print("Prediction probabilities for each digit (0-9):")
        for i, prob in enumerate(probabilities):
            print(f"Digit {i}: {prob:.4f}")

        return predicted_digit
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# 3. Test model on a few MNIST test images to verify behavior
def test_mnist_images():
    print("\nTesting model on MNIST test images...")
    (x_test, y_test) = tf.keras.datasets.mnist.load_data()[1]
    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    for i in range(3):  # Test 3 random images
        img = x_test[i:i+1]
        true_digit = y_test[i]
        pred = model.predict(img)
        pred_digit = np.argmax(pred[0])
        print(f"MNIST test image {i+1}: True digit: {true_digit}, Predicted: {pred_digit}")

# 4. Example usage for custom image prediction
image_path = '../data/raw/3.jpg'  # Your image path
invert_colors = True  # Set to False if your digits are white on black
enhance_contrast = True  # Set to False to disable contrast enhancement

# Test MNIST images first
test_mnist_images()

# Predict on custom image
if os.path.exists(image_path):
    predicted_digit = predict_digit(image_path, invert_colors=invert_colors,
                                  enhance_contrast=enhance_contrast)
    if predicted_digit is not None:
        print(f"The predicted digit is: {predicted_digit}")
else:
    print(f"Image not found: {image_path}. Please check the path.")