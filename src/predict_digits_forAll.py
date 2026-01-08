import tensorflow as tf
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import re

def load_model(model_path='mnist_cnn_model.h5'):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def preprocess_image(image):
    """Enhanced preprocessing pipeline"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 6)
    
    # Morphological operations
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return processed

def extract_digits(processed, model):
    """Robust digit extraction with size validation"""
    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digits = []
    for i, contour in enumerate(contours):
        x,y,w,h = cv2.boundingRect(contour)
        
        # Validate as digit
        if 15 < w < 100 and 20 < h < 100 and (0.3 < w/h < 1.5):
            digit_img = processed[y:y+h, x:x+w]
            
            # Resize to 28x28 with proper padding
            h, w = digit_img.shape
            if w > h:
                new_w = 20
                new_h = int(20 * (h/w))
                padded = np.pad(
                    cv2.resize(digit_img, (new_w, new_h)),
                    ((4,4),(4,4)), 'constant', constant_values=0
                )
            else:
                new_h = 20
                new_w = int(20 * (w/h))
                padded = np.pad(
                    cv2.resize(digit_img, (new_w, new_h)),
                    ((4,4),(4,4)), 'constant', constant_values=0
                )
            
            # Ensure final size is 28x28
            if padded.shape != (28,28):
                padded = cv2.resize(padded, (28,28))
            
            # Normalize and predict
            normalized = padded.astype('float32') / 255.0
            try:
                pred = model.predict(normalized.reshape(1,28,28,1))
                digit = str(np.argmax(pred))
                confidence = np.max(pred)
                
                if confidence > 0.9:
                    digits.append((x, digit))
                    cv2.imwrite(f"digit_{i}_{digit}.png", padded*255)
            except Exception as e:
                print(f"Prediction error for digit {i}: {e}")
                continue
    
    # Sort digits left to right
    digits.sort(key=lambda x: x[0])
    return ''.join([d[1] for d in digits])

def main():
    model = load_model()
    
    # Load and crop account region (using your coordinates)
    image = cv2.imread("../data/raw/slip.jpg")
    account_region = image[int(image.shape[0]/3):int(2*image.shape[0]/3), 50:1600][150:370]
    
    # Preprocess
    processed = preprocess_image(account_region)
    cv2.imwrite("processed_region.png", processed)
    
    # Extract digits
    cnn_result = extract_digits(processed, model)
    
    # OCR verification
    ocr_text = pytesseract.image_to_string(processed, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
    ocr_result = re.sub(r'\D', '', ocr_text)
    
    # Final validation
    final_account = cnn_result if len(cnn_result) >= 8 else ocr_result
    
    print(f"\nCNN Extraction: {cnn_result}")
    print(f"OCR Extraction: {ocr_result}")
    print(f"\n✅ Final Account Number: {final_account}")

if __name__ == "__main__":
    main()