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
    exit(1)

# 2. Function to preprocess and predict a sequence of digits
def predict_digit_sequence(image_path, invert_colors=True, enhance_contrast=True):
    try:
        # Load the image with PIL
        image = Image.open(image_path)
        print(f"Image size: {image.size}")
        
        # Convert to grayscale and enhance
        img = image.convert("L")
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
        img = img.filter(ImageFilter.SHARPEN)
        img_array = np.array(img)
        
        # Thresholding
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to separate digits
        kernel = np.ones((3,3), np.uint8)
        img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find and sort contours (left to right)
        contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        # Filter out small contours (noise)
        min_area = 100
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        print(f"Found {len(contours)} digits in the image")
        
        # Process each digit
        predicted_digits = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding and crop
            padding = 10
            digit = img_array[max(0,y-padding):min(img_array.shape[0],y+h+padding), 
                            max(0,x-padding):min(img_array.shape[1],x+w+padding)]
            
            # Resize to fit in 20x20 while maintaining aspect ratio
            aspect_ratio = digit.shape[1] / digit.shape[0]
            if aspect_ratio > 1:
                new_w = 20
                new_h = int(20 / aspect_ratio)
            else:
                new_h = 20
                new_w = int(20 * aspect_ratio)
            
            digit = cv2.resize(digit, (new_w, new_h))
            
            # Center in 28x28 image
            canvas = np.zeros((28, 28), dtype=np.uint8)
            start_x = (28 - new_w) // 2
            start_y = (28 - new_h) // 2
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = digit
            
            # Normalize
            digit = canvas.astype('float32') / 255.0
            digit = digit.reshape(1, 28, 28, 1)
            
            # Predict
            prediction = model.predict(digit)
            predicted_digit = np.argmax(prediction[0])
            predicted_digits.append(str(predicted_digit))
            
            # Save each digit for debugging
            cv2.imwrite(f'digit_{i}.png', canvas)
            print(f"Digit {i}: Predicted {predicted_digit}")

        return ''.join(predicted_digits)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# 3. Example usage
image_path = '../data/raw/sequencess.jpg'  # Your multi-digit image
if os.path.exists(image_path):
    sequence = predict_digit_sequence(image_path)
    if sequence:
        print(f"\nPredicted digit sequence: {sequence}")
else:
    print("Image not found")