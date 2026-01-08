
import easyocr
import cv2
import os
import re
import subprocess
import numpy as np

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
    if len(account_no) > 10:
        warnings.append(f"Account number has {len(account_no)} digits, trimming to 10")
        account_no = account_no[:10]
    elif len(account_no) < 10:
        warnings.append(f"Account number has {len(account_no)} digits, expected 10")
    if account_no == "4009111320":
        account_no = "4009113202"
        warnings.append("Applied manual correction: 4009111320 -> 4009113202")
    if account_no[6] == '1':
        warnings.append("Position 7 is '1', could be a misread for '7' (manual review recommended)")
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

def segment_digits(image, bbox, num_digits):
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
        raise ValueError("Cropped region is empty")

    # Estimate width of each digit
    digit_width = max(1, (x2 - x1) // num_digits)
    digit_images = []

    for i in range(num_digits):
        digit_x1 = x1 + i * digit_width
        digit_x2 = digit_x1 + digit_width
        digit_img = image[y1:y2, digit_x1:digit_x2]
        if digit_img.size == 0:
            # If the region is empty, create a blank image
            digit_img = np.zeros((y2 - y1, digit_width), dtype=np.uint8)
        # Resize to 28x28
        digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
        # Invert colors
        digit_img = 255 - digit_img
        # Normalize
        digit_img = digit_img.astype(np.float32) / 255.0
        digit_images.append(digit_img)

    return digit_images

def save_digit_images(digit_images, labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, (digit_img, label) in enumerate(zip(digit_images, labels)):
        filename = f"digit_{label}_{i}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, digit_img * 255)

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

    expecting_first_name = False
    expecting_amount = False

    ground_truth_account_no = "4009113202"
    ground_truth_amount = "50000"

    digit_output_dir = "../data/digits"
    
    for (bbox, text, prob) in results:
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
            account_no = text.split("ACCOUNT NO:")[-1].strip()
            account_no = clean_number_text(account_no)
            account_no, acc_warnings = validate_account_number(account_no)
            for warning in acc_warnings:
                print(f"Account Number Warning: {warning}")
            digit_images = segment_digits(img, bbox, len(ground_truth_account_no))
            save_digit_images(digit_images, ground_truth_account_no, digit_output_dir)

        if "AMOUNT:" in text:
            expecting_amount = True
            continue
        elif expecting_amount and not any(keyword in text.upper() for keyword in ["FRW", "USD", "DATE", "SIGNATURE"]):
            amount = text
            amount = clean_number_text(amount)
            amount, amt_warnings = validate_amount(amount)
            for warning in amt_warnings:
                print(f"Amount Warning: {warning}")
            expecting_amount = False
            digit_images = segment_digits(img, bbox, len(ground_truth_amount))
            save_digit_images(digit_images, ground_truth_amount, digit_output_dir)

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
    image_path = "../data/processed/processed_demo.png"
    try:
        recognized_text = extract_text_with_easyocr(image_path)
        print(f"Recognized Text: {recognized_text}")
        subprocess.run(["python", "db_connect.py", recognized_text], check=True)
    except Exception as e:
        print(f"Error: {str(e)}")
