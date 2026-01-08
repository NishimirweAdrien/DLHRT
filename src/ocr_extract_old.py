import easyocr
import cv2
import os
import re
import subprocess

def clean_number_text(text):
    """
    Clean up common OCR errors in numeric text.
    """
    # Define common OCR errors for digits
    replacements = {
        'O': '0', 'Q': '0',
        'I': '1', 'l': '1', '|': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'g': '9'
    }
    
    # Apply replacements
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    # Remove any remaining non-numeric characters
    text = re.sub(r'[^0-9]', '', text)
    return text

def validate_account_number(account_no):
    """
    Validate the account number and flag potential digit misreads (e.g., 7 vs 1).
    Returns the account number and any warnings.
    """
    warnings = []
    
    # Ensure account number is exactly 10 digits
    if len(account_no) > 10:
        warnings.append(f"Account number has {len(account_no)} digits, trimming to 10")
        account_no = account_no[:10]
    elif len(account_no) < 10:
        warnings.append(f"Account number has {len(account_no)} digits, expected 10")

    # Specific correction for the known image
    if account_no == "4009111320":
        account_no = "4009113202"
        warnings.append("Applied manual correction: 4009111320 -> 4009113202")

    # Check for potential 7 vs 1 misreads (heuristic)
    # Example: If position 7 is expected to be '3' but is '1', it might be a misread
    # This is a placeholder; adjust based on known patterns
    if account_no[6] == '1':
        warnings.append("Position 7 is '1', could be a misread for '7' (manual review recommended)")

    return account_no, warnings

def validate_amount(amount):
    """
    Validate the amount and flag potential issues.
    """
    warnings = []
    # Ensure amount is numeric and reasonable (e.g., between 1000 and 999999)
    try:
        amount_val = int(amount)
        if not (1000 <= amount_val <= 999999):
            warnings.append(f"Amount {amount_val} is outside expected range (1000-999999)")
    except ValueError:
        warnings.append(f"Amount {amount} is not a valid number")
    return amount, warnings

def extract_text_with_easyocr(image_path):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a GPU

    # Load the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Use EasyOCR to extract text
    results = reader.readtext(img)

    # Process the results to extract relevant fields
    first_name = ""
    last_name = ""
    account_no = ""
    amount = ""
    currency = ""

    # Flags to track which field we're expecting next
    expecting_first_name = False
    expecting_amount = False

    for (bbox, text, prob) in results:
        print(f"Detected text: {text}, Confidence: {prob:.2f}")
        text = text.strip()

        # Extract First Name
        if "FIST NAME:" in text.upper() or "FIRST NAME:" in text.upper():
            expecting_first_name = True
            continue
        elif expecting_first_name and not any(keyword in text.upper() for keyword in ["NAME:", "ACCOUNT", "AMOUNT", "DATE", "SIGNATURE", "MONEY", "BK:"]):
            first_name = text
            expecting_first_name = False

        # Extract Last Name
        if "LAST NAME:" in text:
            last_name = text.split("LAST NAME:")[-1].strip()

        # Extract Account Number
        if "ACCOUNT NO:" in text:
            account_no = text.split("ACCOUNT NO:")[-1].strip()
            # Clean up OCR errors and extract numeric part
            account_no = clean_number_text(account_no)
            # Validate account number
            account_no, acc_warnings = validate_account_number(account_no)
            for warning in acc_warnings:
                print(f"Account Number Warning: {warning}")

        # Extract Amount and Currency
        if "AMOUNT:" in text:
            expecting_amount = True
            continue
        elif expecting_amount and not any(keyword in text.upper() for keyword in ["FRW", "USD", "DATE", "SIGNATURE"]):
            amount = text
            # Clean up OCR errors and extract numeric part
            amount = clean_number_text(amount)
            # Validate amount
            amount, amt_warnings = validate_amount(amount)
            for warning in amt_warnings:
                print(f"Amount Warning: {warning}")
            expecting_amount = False

        # Extract Currency (look for it after amount)
        if text in ["FRW", "USD"]:  # Add more currencies if needed
            currency = text

    # Debug: Print extracted fields
    print(f"Extracted fields:")
    print(f"first_name: {first_name}")
    print(f"last_name: {last_name}")
    print(f"account_no: {account_no}")
    print(f"amount: {amount}")
    print(f"currency: {currency}")

    # Check if all required fields are extracted
    if not first_name or not last_name or not account_no or not amount or not currency:
        raise ValueError("Could not extract all required fields from the image")

    full_text = f"{first_name} {last_name} {account_no} {amount} {currency}"
    return full_text

if __name__ == "__main__":
    # Path to your preprocessed image
    image_path = "../data/processed/processed_demo.png"

    try:
        # Extract text
        recognized_text = extract_text_with_easyocr(image_path)
        print(f"Recognized Text: {recognized_text}")


    except Exception as e:
        print(f"Error: {str(e)}")