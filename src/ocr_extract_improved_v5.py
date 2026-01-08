import cv2
import numpy as np
import os
import tensorflow as tf

# Function to preprocess the image for digit recognition
def preprocess_image_for_digits(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=2)  # Increased dilation
    binary = cv2.erode(binary, kernel, iterations=1)  # Separate digits
    return binary

# Function to segment digits in the value region
def segment_digits(image, bbox, field_name, expected_digits=10):
    try:
        print(f"Segmenting digits for {field_name}...")
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        digit_region = image[y1:y2, x1:x2]
        
        _, binary = cv2.threshold(digit_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.erode(binary, kernel, iterations=2)  # Increased erosion to separate digits
        
        debug_dir = "../data/debug"
        cv2.imwrite(os.path.join(debug_dir, f"{field_name}_binary.png"), binary)
        print(f"Saved binary image for {field_name} to {os.path.join(debug_dir, f'{field_name}_binary.png')}")

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} contours for {field_name}")

        digit_images = []
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}")
            if w > 5 and h > 5 and w < digit_region.shape[1] * 0.5 and h < digit_region.shape[0] * 1.0:
                digit = binary[y:y+h, x:x+w]
                top = max(0, (28 - h) // 2)
                bottom = max(0, 28 - h - top)
                left = max(0, (28 - w) // 2)
                right = max(0, 28 - w - left)
                digit = cv2.copyMakeBorder(digit, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
                digit = cv2.bitwise_not(digit)
                digit_images.append(digit)
                cv2.imwrite(os.path.join(debug_dir, f"{field_name}_digit_{i}.png"), digit)
                print(f"Saved digit {i} for {field_name} to {os.path.join(debug_dir, f'{field_name}_digit_{i}.png')}")
            else:
                print(f"Rejected contour {i}: w={w} (max {digit_region.shape[1]*0.5}), h={h} (max {digit_region.shape[0]*1.0})")
                rejected_digit = binary[y:y+h, x:x+w]
                if rejected_digit.shape[0] > 0 and rejected_digit.shape[1] > 0:
                    rejected_digit = cv2.resize(rejected_digit, (28, 28), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(debug_dir, f"{field_name}_rejected_digit_{i}.png"), rejected_digit)
                    print(f"Saved rejected contour {i} for {field_name} to {os.path.join(debug_dir, f'{field_name}_rejected_digit_{i}.png')}")

        if len(digit_images) < expected_digits * 0.7:
            print(f"Contours insufficient ({len(digit_images)}), falling back to grid-based segmentation")
            digit_images = []
            region_width = digit_region.shape[1]
            digit_width = region_width // expected_digits
            for i in range(expected_digits):
                start_x = i * digit_width
                end_x = (i + 1) * digit_width
                digit = binary[:, start_x:end_x]
                if digit.shape[1] > 0:
                    digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
                    digit = cv2.bitwise_not(digit)
                    digit_images.append(digit)
                    cv2.imwrite(os.path.join(debug_dir, f"{field_name}_digit_{i}.png"), digit)
                    print(f"Saved digit {i} for {field_name} to {os.path.join(debug_dir, f'{field_name}_digit_{i}.png')}")

        if not digit_images:
            raise ValueError(f"No digits segmented for {field_name}. Check the bounding box and preprocessing.")

        return digit_images
    except Exception as e:
        print(f"Error in segment_digits: {str(e)}")
        raise

# Function to recognize digits using your model
def recognize_digits(digit_images, field_name, model):
    try:
        print(f"Recognizing digits for {field_name}...")
        digits = []
        for i, digit in enumerate(digit_images):
            digit = digit.astype('float32') / 255.0
            digit = np.expand_dims(digit, axis=(0, -1))
            prediction = model.predict(digit)
            digit_value = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            print(f"{field_name} digit {i}: Predicted {digit_value} with confidence {confidence:.4f}")
            digits.append(str(digit_value))
        combined = ''.join(digits)
        print(f"Combined digits for {field_name}: {combined}")
        return combined
    except Exception as e:
        print(f"Error in recognize_digits: {str(e)}")
        raise

# Main function to extract the ACCOUNT NO value
def extract_account_no_value(image_path, model_path):
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

        # Manually specify the bounding box for the ACCOUNT NO value
        # Replace with the coordinates from the coordinate finder script
        account_no_value_bbox = [[500, 283], [700, 283], [700, 311], [500, 311]]  # Update these

        print(f"ACCOUNT NO value region: {account_no_value_bbox}")

        # Preprocess the image for digit recognition
        print("Preprocessing image for digit recognition...")
        img_digits = preprocess_image_for_digits(img)
        debug_dir = "../data/debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "preprocessed_digits_binary.png"), img_digits)
        print(f"Saved digit preprocessed image to {os.path.join(debug_dir, 'preprocessed_digits_binary.png')}")

        # Extract the ACCOUNT NO value
        digit_images = segment_digits(img_digits, account_no_value_bbox, "account_no_value")
        account_no_value = recognize_digits(digit_images, "account_no_value", digit_model)
        
        print(f"\nExtracted ACCOUNT NO value: {account_no_value}")
        
        # Save the extracted value to a plain text file
        with open("../data/extracted_account_no.txt", "w") as f:
            f.write(account_no_value)
        print(f"Saved extracted value to ../data/extracted_account_no.txt")

        return account_no_value

    except Exception as e:
        print(f"Error in extract_account_no_value: {str(e)}")
        raise

if __name__ == "__main__":
    image_path = "../data/processed/processed_demod.jpg"
    model_path = "../models/digit_cnn_mnist.h5"
    account_no_value = extract_account_no_value(image_path, model_path)