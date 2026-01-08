
import easyocr
import cv2
import os
import re
import subprocess
import numpy as np
import mysql.connector
from tensorflow.keras.models import load_model

# Load the trained CNN model
digit_model = load_model("../models/digit_cnn.h5")

def setup_database():
    """
    Set up the MySQL database connection.
    
    Returns:
        mysql.connector.connection.MySQLConnection: Database connection object.
    """
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="banking_system"
        )
        if conn.is_connected():
            print("Successfully connected to MySQL database")
        return conn
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        raise

def save_to_database(conn, first_name, surname, account_number, amount):
    """
    Save the extracted form data to the MySQL database.
    
    Args:
        conn (mysql.connector.connection.MySQLConnection): Database connection object.
        first_name (str): Extracted first name.
        surname (str): Extracted surname.
        account_number (str): Extracted account number.
        amount (str): Extracted amount.
    """
    try:
        cursor = conn.cursor()
        # Insert the extracted data into the customers table
        query = "INSERT INTO customers (first_name, surname, account_no, amount) VALUES (%s, %s, %s, %s)"
        values = (first_name, surname, account_number, amount)
        cursor.execute(query, values)
        conn.commit()
        print(f"Saved to database: First Name={first_name}, Surname={surname}, Account Number={account_number}, Amount={amount}")
    except mysql.connector.Error as e:
        print(f"Database save error: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def preprocess_image_for_ocr(image):
    """
    Preprocess the image to enhance handwritten text for OCR.
    
    Args:
        image (numpy.ndarray): Grayscale input image.
    
    Returns:
        numpy.ndarray: Preprocessed image.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Enhance contrast using adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Apply adaptive thresholding to binarize the image
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 3
    )
    
    return binary

def clean_number_text(text):
    replacements = {
        'O': '0', 'Q': '0',
        'I': '1', 'l': '1', '|': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'g': '9'
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    text = re.sub(r'[^0-9]', '', text)
    return text

def validate_account_number(account_no):
    warnings = []
    if len(account_no) > 11:
        warnings.append(f"Account number has {len(account_no)} digits, trimming to 11")
        account_no = account_no[:11]
    elif len(account_no) < 11:
        warnings.append(f"Account number has {len(account_no)} digits, expected 11")
    return account_no, warnings

def validate_amount(amount):
    warnings = []
    try:
        amount_val = int(amount)
        if not (1000 <= amount_val <= 999999):
            warnings.append(f"Amount {amount_val} is outside expected range (1000-999999)")
    except ValueError:
        warnings.append(f"Amount {amount} is not a valid number")
    return amount, warnings

def segment_digits(image, bbox, num_digits, field_name, debug_dir="../data/debug"):
    top_left, top_right, bottom_right, bottom_left = bbox
    x1, y1 = int(top_left[0]), int(top_left[1])
    x2, y2 = int(bottom_right[0]), int(bottom_right[1])
    
    # Ensure coordinates are valid
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")

    # Crop the region
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        raise ValueError(f"Cropped region is empty")

    # Preprocess the region for handwritten digits
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply erosion to separate digits (instead of dilation)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    # Save the binary image for debugging
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    debug_filepath = os.path.join(debug_dir, f"{field_name}_binary.png")
    cv2.imwrite(debug_filepath, binary)
    print(f"Saved binary image for {field_name} to {debug_filepath}")

    # Find contours to identify individual digits
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Sort by x-coordinate
    
    print(f"Found {len(contours)} contours for {field_name}")
    
    digit_images = []
    if len(contours) >= num_digits:
        # Use contours if enough are found
        for i in range(num_digits):
            x, y, w, h = cv2.boundingRect(contours[i])
            print(f"Digit {i} bounding box: x={x}, y={y}, w={w}, h={h}")
            digit_img = binary[y:y+h, x:x+w]
            if digit_img.size == 0:
                digit_img = np.zeros((28, 28), dtype=np.uint8)
            else:
                # Pad the digit image to ensure proper resizing
                digit_img = cv2.copyMakeBorder(
                    digit_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0
                )
                digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
            digit_img = digit_img.astype(np.float32) / 255.0
            digit_images.append(digit_img)
    else:
        # Fallback to grid-based segmentation
        print(f"Contours insufficient ({len(contours)}), falling back to grid-based segmentation")
        region_width = x2 - x1
        digit_width = max(1, region_width // num_digits)
        for i in range(num_digits):
            digit_x1 = x1 + i * digit_width
            digit_x2 = digit_x1 + digit_width
            digit_img = image[y1:y2, digit_x1:digit_x2]
            if digit_img.size == 0:
                digit_img = np.zeros((y2 - y1, digit_width), dtype=np.uint8)
            # Preprocess each digit region
            _, digit_binary = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            digit_binary = cv2.erode(digit_binary, kernel, iterations=1)
            digit_img = cv2.resize(digit_binary, (28, 28), interpolation=cv2.INTER_AREA)
            digit_img = digit_img.astype(np.float32) / 255.0
            # Save each digit for debugging
            debug_digit_filepath = os.path.join(debug_dir, f"{field_name}_digit_{i}.png")
            cv2.imwrite(debug_digit_filepath, digit_img * 255)
            print(f"Saved digit {i} for {field_name} to {debug_digit_filepath}")
            digit_images.append(digit_img)
    
    return digit_images

def recognize_digits(digit_images, field_name, output_dir="../data/inference_digits"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    digits = []
    for i, digit_img in enumerate(digit_images):
        # Recognize one digit at a time
        digit_img_np = np.array(digit_img).reshape(-1, 28, 28, 1)
        prediction = digit_model.predict(digit_img_np)
        digit = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]
        filename = f"{field_name}_digit_{i}_pred_{digit}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, digit_img * 255)
        print(f"{field_name} digit {i}: Predicted {digit} with confidence {confidence:.4f}")
        digits.append(str(digit))
    
    # Combine digits into a single string without spaces
    return ''.join(digits)

def extract_text_with_easyocr(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Preprocess the image for better handwritten text detection
    img_processed = preprocess_image_for_ocr(img)
    
    # Save the preprocessed image for debugging
    debug_dir = "../data/debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    cv2.imwrite(os.path.join(debug_dir, "preprocessed_image.png"), img_processed)
    print("Saved preprocessed image to ../data/debug/preprocessed_image.png")

    # Use EasyOCR with parameters tuned for handwritten text
    results = reader.readtext(
        img_processed,
        paragraph=False,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        allowlist=None
    )

    first_name = ""
    last_name = ""
    account_no = ""
    amount = ""
    currency = ""
    raw_account_no = ""
    raw_amount = ""

    expecting_first_name = False
    expecting_amount = False
    expecting_account_no = False
    account_no_bbox = None
    amount_bbox = None

    # Store all results with their bounding boxes for proximity checks
    text_with_bboxes = [(bbox, text, prob) for bbox, text, prob in results]

    for i, (bbox, text, prob) in enumerate(text_with_bboxes):
        # Remove underscore only for "ACCOUNT NO:" field
        if "ACCOUNT NO:" in text:
            text = text.replace('_', ' ')
        print(f"Detected text: {text}, Confidence: {prob:.2f}, BBox: {bbox}")
        text = text.strip()

        if "FIST NAME:" in text.upper() or "FIRST NAME:" in text.upper() or "FSTNAIE" in text.upper():
            expecting_first_name = True
            print(f"Expecting first name after: {text}")
            continue
        elif expecting_first_name:
            # Check the current or next text block for proximity
            current_y = bbox[0][1]
            if i < len(text_with_bboxes):
                # Check current block
                if not any(keyword in text.upper() for keyword in ["LAST", "ACCOUNT", "AMOUNT", "DATE", "SIGNATURE"]):
                    first_name = text
                    print(f"Assigned first_name from current block: {first_name}")
                    expecting_first_name = False
                # Check next block
                elif i + 1 < len(text_with_bboxes):
                    next_bbox, next_text, next_prob = text_with_bboxes[i + 1]
                    next_y = next_bbox[0][1]
                    if abs(next_y - current_y) < 150:  # Further relaxed threshold
                        first_name = next_text
                        print(f"Assigned first_name from next block: {first_name}")
                        expecting_first_name = False

        if "LAST NAME:" in text.upper() or "LASTNAME" in text.upper() or "LASTNANIE" in text.upper():
            expecting_last_name = True
            last_name = text.split("LAST NAME:")[-1].strip() or ""
            if not last_name and i + 1 < len(text_with_bboxes):
                # Check the next text block for proximity
                next_bbox, next_text, next_prob = text_with_bboxes[i + 1]
                current_y = bbox[0][1]
                next_y = next_bbox[0][1]
                if abs(next_y - current_y) < 150:  # Further relaxed threshold
                    last_name = next_text
                    print(f"Assigned last_name: {last_name}")
            else:
                print(f"No last name found in same block: {text}")
            expecting_last_name = False

        if "ACCOUNT NO:" in text.upper() or "ACCOUNTNO" in text.upper():
            expecting_account_no = True
            raw_account_no = text.split("ACCOUNT NO:")[-1].strip()
            account_no_bbox = bbox
            # If the account number is in the same block, process it immediately
            cleaned_account_no = clean_number_text(raw_account_no)
            if cleaned_account_no:
                digit_images = segment_digits(img_processed, account_no_bbox, 11, "account_no")
                account_no = recognize_digits(digit_images, "account_no")
                print(f"Digit recognition result for account number: {account_no}")
                print(f"Using digit recognition for account number (OCR text: {raw_account_no})")
                account_no, acc_warnings = validate_account_number(account_no)
                for warning in acc_warnings:
                    print(f"Account Number Warning: {warning}")
                expecting_account_no = False
            continue
        elif expecting_account_no:
            # Check the next text block for proximity
            if i + 1 < len(text_with_bboxes):
                next_bbox, next_text, next_prob = text_with_bboxes[i + 1]
                current_y = bbox[0][1]
                next_y = next_bbox[0][1]
                if abs(next_y - current_y) < 150:
                    raw_account_no = next_text
                    digit_images = segment_digits(img_processed, next_bbox, 11, "account_no")
                    account_no = recognize_digits(digit_images, "account_no")
                    print(f"Digit recognition result for account number: {account_no}")
                    print(f"Using digit recognition for account number (OCR text: {raw_account_no})")
                    account_no, acc_warnings = validate_account_number(account_no)
                    for warning in acc_warnings:
                        print(f"Account Number Warning: {warning}")
                    expecting_account_no = False

        if "AMOUNT:" in text.upper() or "AMOUNT" in text.upper():
            expecting_amount = True
            amount_bbox = bbox
            print(f"Expecting amount after: {text}")
            continue
        elif expecting_amount:
            # Check the next text block for proximity
            if i + 1 < len(text_with_bboxes):
                next_bbox, next_text, next_prob = text_with_bboxes[i + 1]
                current_y = amount_bbox[0][1]
                next_y = next_bbox[0][1]
                if abs(next_y - current_y) < 150:
                    raw_amount = next_text
                    digit_images = segment_digits(img_processed, next_bbox, 5, "amount")
                    amount = recognize_digits(digit_images, "amount")
                    print(f"Digit recognition result for amount: {amount}")
                    print(f"Using digit recognition for amount (OCR text: {raw_amount})")
                    amount, amt_warnings = validate_amount(amount)
                    for warning in amt_warnings:
                        print(f"Amount Warning: {warning}")
                    expecting_amount = False

        if text in ["FRW", "USD", "FRI"]:
            currency = "FRW" if text == "FRI" else text

    print(f"Extracted fields:")
    print(f"first_name: {first_name}")
    print(f"last_name: {last_name}")
    print(f"account_no: {account_no}")
    print(f"amount: {amount}")
    print(f"currency: {currency}")

    if not first_name or not last_name or not account_no or not amount or not currency:
        raise ValueError("Could not extract all required fields from the image")

    full_text = f"{first_name} {last_name} {account_no} {amount} {currency}"
    return first_name, last_name, account_no, amount, full_text

if __name__ == "__main__":
    image_path = "../data/processed/processed_demod.jpg"
    try:
        # Set up database connection
        conn = setup_database()
        
        # Extract text from image
        first_name, last_name, account_no, amount, recognized_text = extract_text_with_easyocr(image_path)
        print(f"Recognized Text: {recognized_text}")
        
        # Save to database
        save_to_database(conn, first_name, last_name, account_no, amount)
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Close the database connection
        if 'conn' in locals() and conn.is_connected():
            conn.close()
            print("Database connection closed")
