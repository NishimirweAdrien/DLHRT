import cv2
import numpy as np
import os
import tensorflow as tf

# Function to preprocess the image for digit detection
def preprocess_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    debug_dir = "../data/debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    cv2.imwrite(os.path.join(debug_dir, "original.png"), image)
    print(f"Saved original image to {os.path.join(debug_dir, 'original.png')}")

    # Resize the image to a manageable size (e.g., 512x384)
    height, width = image.shape
    scale_factor = 512 / height
    new_width = int(width * scale_factor)
    new_height = 512
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(debug_dir, "resized.png"), image)
    print(f"Saved resized image to {os.path.join(debug_dir, 'resized.png')}")

    # Denoise the image
    image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(os.path.join(debug_dir, "blurred.png"), image)
    print(f"Saved blurred image to {os.path.join(debug_dir, 'blurred.png')}")

    # Try Otsu's thresholding first
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(debug_dir, "binary_otsu.png"), binary)
    print(f"Saved Otsu binary image to {os.path.join(debug_dir, 'binary_otsu.png')}")

    # If Otsu fails, fall back to adaptive thresholding
    if cv2.countNonZero(binary) < 100:  # Check if the binary image is mostly black
        print("Otsu thresholding produced an empty image, trying adaptive thresholding...")
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        cv2.imwrite(os.path.join(debug_dir, "binary_adaptive.png"), binary)
        print(f"Saved adaptive binary image to {os.path.join(debug_dir, 'binary_adaptive.png')}")

    # Morphological operations to remove noise and connect digit parts
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.erode(binary, kernel, iterations=1)  # Reduced erosion
    binary = cv2.dilate(binary, kernel, iterations=2)  # Dilate to connect digit parts
    cv2.imwrite(os.path.join(debug_dir, "binary_morph.png"), binary)
    print(f"Saved morphed binary image to {os.path.join(debug_dir, 'binary_morph.png')}")

    return binary

# Function to segment a single digit in the image
def segment_single_digit(image):
    try:
        print("Segmenting digit...")
        debug_dir = "../data/debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "binary.png"), image)
        print(f"Saved final binary image to {os.path.join(debug_dir, 'binary.png')}")

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours")

        digit_images = []
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        # Sort by area (largest first) to prioritize the actual digit
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[2] * b[3], reverse=True)
        
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}")
            # Tight filtering criteria for a single digit
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            if (w > 20 and h > 20 and  # Minimum size
                w < image.shape[1] * 0.3 and h < image.shape[0] * 0.5 and  # Maximum size
                0.3 < aspect_ratio < 1.5 and  # Aspect ratio for digits
                area > 500):  # Minimum area to reject noise
                digit = image[y:y+h, x:x+w]
                top = max(0, (28 - h) // 2)
                bottom = max(0, 28 - h - top)
                left = max(0, (28 - w) // 2)
                right = max(0, 28 - w - left)
                digit = cv2.copyMakeBorder(digit, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
                digit = cv2.bitwise_not(digit)
                digit_images.append(digit)
                cv2.imwrite(os.path.join(debug_dir, f"digit_{i}.png"), digit)
                print(f"Saved digit {i} to {os.path.join(debug_dir, f'digit_{i}.png')}")
                break  # Stop after finding the first valid digit
            else:
                print(f"Rejected contour {i}: w={w} (max {image.shape[1]*0.3}), h={h} (max {image.shape[0]*0.5}), aspect_ratio={aspect_ratio}, area={area}")
                rejected_digit = image[y:y+h, x:x+w]
                if rejected_digit.shape[0] > 0 and rejected_digit.shape[1] > 0:
                    rejected_digit = cv2.resize(rejected_digit, (28, 28), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(debug_dir, f"rejected_digit_{i}.png"), rejected_digit)
                    print(f"Saved rejected contour {i} to {os.path.join(debug_dir, f'rejected_digit_{i}.png')}")

        if not digit_images:
            raise ValueError("No digit segmented. Check the image and preprocessing.")

        if len(digit_images) > 1:
            print(f"Warning: Expected 1 digit, but found {len(digit_images)}. Using the first digit.")

        return digit_images[:1]  # Return only the first digit
    except Exception as e:
        print(f"Error in segment_single_digit: {str(e)}")
        raise

# Function to recognize a single digit using the model
def recognize_digit(digit_image, model):
    try:
        print("Recognizing digit...")
        digit = digit_image.astype('float32') / 255.0
        digit = np.expand_dims(digit, axis=(0, -1))
        prediction = model.predict(digit)
        digit_value = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        print(f"Predicted {digit_value} with confidence {confidence:.4f}")
        return str(digit_value)
    except Exception as e:
        print(f"Error in recognize_digit: {str(e)}")
        raise

# Main function to detect a single digit in an image
def detect_single_digit(image_path, model_path):
    try:
        print("Loading digit recognition model...")
        digit_model = tf.keras.models.load_model(model_path)
        print("Digit recognition model loaded successfully.")

        print("Loading image...")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        print("Image loaded successfully.")
        print(f"Image dimensions: {img.shape}")

        # Preprocess the image
        print("Preprocessing image...")
        img_processed = preprocess_image(img)

        # Segment the digit
        digit_images = segment_single_digit(img_processed)

        # Recognize the digit
        detected_digit = recognize_digit(digit_images[0], digit_model)
        
        print(f"\nDetected digit: {detected_digit}")
        
        # Save the detected digit to a plain text file
        with open("../data/detected_digit.txt", "w") as f:
            f.write(detected_digit)
        print(f"Saved detected digit to ../data/detected_digit.txt")

        return detected_digit

    except Exception as e:
        print(f"Error in detect_single_digit: {str(e)}")
        raise

if __name__ == "__main__":
    image_path = "../data/raw/2.jpg"
    model_path = "../models/digit_cnn_mnist_two.h5"
    detected_digit = detect_single_digit(image_path, model_path)