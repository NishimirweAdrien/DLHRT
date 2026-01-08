
import easyocr
import cv2
import os
import re
import subprocess
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained CNN model
digit_model = load_model("../models/digit_cnn.h5")

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
    x2, y2 = int(bottom_right[0]), int(bottom_right[1])  # Fixed typo: custom_right -> bottom_right
    
    # Ensure coordinates are valid
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")

    # Crop the region
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        raise ValueError("Cropped region is empty")

    # Binarize the image
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Save the binary image for debugging
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    debug_filepath = os.path.join(debug_dir, f"{field_name}_binary.png")
    cv2.imwrite(debug_filepath, binary)
    print(f"Saved binary image for {field_name} to {debug_filepath}")

    # Use equal spacing to segment digits, with improved alignment
    region_width = x2 - x1
    digit_width = max(1, region_width // num_digits)  # Base width for each digit
    digit_images = []
    # Estimate digit positions by finding approximate centers
    total_width = x2 - x1
    spacing = total_width / (num_digits + 1)  # Space between digit centers
    for i in range(num_digits):
        # Calculate the center of the digit
        center_x = x1 + (i + 1) * spacing
        digit_x1 = int(center_x - digit_width // 2)
        digit_x2 = int(center_x + digit_width // 2)
        digit_img = image[y1:y2, digit_x1:digit_x2]
        if digit_img.size == 0:
            digit_img = np.zeros((y2 - y1, digit_width), dtype=np.uint8)
        digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
        digit_img = 255 - digit_img
        digit_img = digit_img.astype(np.float32) / 255.0
        digit_images.append(digit_img)
    return digit_images

def recognize_digits(digit_images, field_name, output_dir="../data/inference_digits"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    digit_images_np = np.array(digit_images).reshape(-1, 28, 28, 1)
    predictions = digit_model.predict(digit_images_np)
    digits = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    for i, (digit_img, digit, confidence) in enumerate(zip(digit_images, digits, confidences)):
        filename = f"{field_name}_digit_{i}_pred_{digit}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, digit_img * 255)
        print(f"{field_name} digit {i}: Predicted {digit} with confidence {confidence:.4f}")

    return ''.join(map(str, digits))

def extract_text_with_easyocr(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    results = reader.readtext(img)

    first_name = ""
    last_name = ""
    account_no = ""
    amount = ""
    currency = ""
    raw_account_no = ""
    raw_amount = ""

    expecting_first_name = False
    expecting_amount = False

    for (bbox, text, prob) in results:
        # Remove underscore only for "ACCOUNT NO:" field
        if "ACCOUNT NO:" in text:
            text = text.replace('_', ' ')
        print(f"Detected text: {text}, Confidence: {prob:.2f}")
        text = text.strip()

        if "FIST NAME:" in text.upper() or "FIRST NAME:" in text.upper():
            expecting_first_name = True
            continue
        elif expecting_first_name and not any(keyword in text.upper() for keyword in ["NAME:", "ACCOUNT", "AMOUNT", "DATE", "SIGNATURE", "MONEY", "BK:"]):
            first_name = text
            expecting_first_name = False

        if "LAST NAME:" in text:
            last_name = text.split("LAST NAME:")[-1].strip()

        if "ACCOUNT NO:" in text:
            raw_account_no = text.split("ACCOUNT NO:")[-1].strip()
            digit_images = segment_digits(img, bbox, 11, "account_no")  # Expect 11 digits
            account_no = recognize_digits(digit_images, "account_no")
            # Fallback to cleaned OCR text
            print(f"Digit recognition result for account number: {account_no}")
            print(f"Using cleaned OCR text for account number: {raw_account_no}")
            account_no = clean_number_text(raw_account_no)
            account_no, acc_warnings = validate_account_number(account_no)
            for warning in acc_warnings:
                print(f"Account Number Warning: {warning}")

        if "AMOUNT:" in text:
            expecting_amount = True
            continue
        elif expecting_amount and not any(keyword in text.upper() for keyword in ["FRW", "USD", "DATE", "SIGNATURE"]):
            raw_amount = text
            digit_images = segment_digits(img, bbox, 5, "amount")
            amount = recognize_digits(digit_images, "amount")
            # Fallback to cleaned OCR text
            print(f"Digit recognition result for amount: {amount}")
            print(f"Using cleaned OCR text for amount: {raw_amount}")
            amount = clean_number_text(raw_amount)
            amount, amt_warnings = validate_amount(amount)
            for warning in amt_warnings:
                print(f"Amount Warning: {warning}")
            expecting_amount = False

        if text in ["FRW", "USD"]:
            currency = text

    print(f"Extracted fields:")
    print(f"first_name: {first_name}")
    print(f"last_name: {last_name}")
    print(f"account_no: {account_no}")
    print(f"amount: {amount}")
    print(f"currency: {currency}")

    if not first_name or not last_name or not account_no or not amount or not currency:
        raise ValueError("Could not extract all required fields from the image")

    full_text = f"{first_name} {last_name} {account_no} {amount} {currency}"
    return full_text

if __name__ == "__main__":
    image_path = "../data/processed/processed_demo6.png"
    try:
        recognized_text = extract_text_with_easyocr(image_path)
        print(f"Recognized Text: {recognized_text}")
    except Exception as e:
        print(f"Error: {str(e)}")
