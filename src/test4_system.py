import cv2
import numpy as np
import pytesseract
import easyocr
import re
import mysql.connector
from keras.models import load_model

# Load the CNN digit recognition model
model = load_model("digit_recognition_cnn_finetuned.h5")

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

def setup_database():
    """Set up MySQL database connection"""
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="banking_system"
        )
        if conn.is_connected():
            print("✅ Successfully connected to MySQL database")
        return conn
    except mysql.connector.Error as e:
        print(f"❌ Database connection error: {e}")
        raise

def get_client_names(conn, account_number):
    """Retrieve first_name and last_name from clients table based on account_no"""
    try:
        cursor = conn.cursor()
        query = "SELECT first_name, last_name FROM clients WHERE account_no = %s"
        cursor.execute(query, (account_number,))
        result = cursor.fetchone()
        if result:
            print(f"✅ Found client for account number {account_number}")
            return result[0], result[1]  # Return first_name, last_name
        else:
            print(f"❌ No client found for account number {account_number}")
            return None, None
    except mysql.connector.Error as e:
        print(f"❌ Error querying clients table: {e}")
        return None, None
    finally:
        if cursor:
            cursor.close()

def save_to_database(conn, first_name, surname, account_number, amount):
    """Save extracted data to MySQL"""
    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO customers (first_name, surname, account_no, amount) 
        VALUES (%s, %s, %s, %s)
        """
        values = (first_name, surname, account_number, amount)
        cursor.execute(query, values)
        conn.commit()
        print(f"💾 Saved to DB: {first_name} {surname} | Acc: {account_number} | Amt: {amount} FRW")
    except mysql.connector.Error as e:
        print(f"❌ Database save error: {e}")
        conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()

# Load the full slip image
image_path = "../data/raw/3_page-0001.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Initialize variables for extracted attributes
account_number = ""
amount = ""
first_name = None
last_name = None
found_account = False
found_amount = False

# --- Account Number and Amount Detection ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh_ocr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# OCR to extract word locations using Tesseract
data = pytesseract.image_to_data(thresh_ocr, output_type=pytesseract.Output.DICT)

for i, word in enumerate(data['text']):
    word_lower = word.lower()

    # Account Number Detection
    if not found_account and 'account' in word_lower:
        print(f"✅ Found 'account' at index {i}")
        for j in range(i + 1, len(data['text'])):
            if re.search(r'\d', data['text'][j]):
                x = data['left'][j]
                y = data['top'][j]
                h = data['height'][j]
                manual_width = 900
                pad = 15

                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + manual_width, image.shape[1])
                y2 = y + h + pad

                cropped = gray[y1:y2, x1:x2]
                _, thresh_account = cv2.threshold(cropped, 150, 255, cv2.THRESH_BINARY_INV)

                contours, _ = cv2.findContours(thresh_account, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                digit_boxes = []
                for cnt in contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    if bw > 5 and bh > 20:
                        digit_boxes.append((bx, by, bw, bh))

                digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

                for (bx, by, bw, bh) in digit_boxes:
                    digit_img = thresh_account[by:by+bh, bx:bx+bw]
                    digit_img = cv2.resize(digit_img, (28, 28))
                    digit_img = digit_img.astype("float32") / 255.0
                    digit_img = np.expand_dims(digit_img, axis=-1)
                    digit_img = np.expand_dims(digit_img, axis=0)

                    prediction = model.predict(digit_img, verbose=0)
                    digit = np.argmax(prediction)
                    account_number += str(digit)

                found_account = True
                break

    # Amount Detection
    if not found_amount and 'amount' in word_lower:
        print(f"✅ Found 'amount' at index {i}")
        for j in range(i + 1, len(data['text'])):
            if re.search(r'\d', data['text'][j]):
                x = data['left'][j]
                y = data['top'][j]
                h = data['height'][j]
                manual_width = 360  # Increase if numbers are long
                pad = 10

                x1 = max(x - 50, 0)
                y1 = max(y - 80,0)
                x2 = min(x + manual_width, image.shape[1])
                y2 = y + h + pad


                amount_crop = image[y1:y2, x1:x2]
                gray_amount = cv2.cvtColor(amount_crop, cv2.COLOR_BGR2GRAY)
                _, amount_thresh = cv2.threshold(gray_amount, 150, 255, cv2.THRESH_BINARY_INV)

                # Apply dilation with smaller kernel to thicken digits
                kernel = np.ones((2, 2), np.uint8)
                amount_thresh = cv2.dilate(amount_thresh, kernel, iterations=1)

                contours, _ = cv2.findContours(amount_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                digit_boxes = []
                for cnt in contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    if bw > 5 and bh > 20:
                        digit_boxes.append((bx, by, bw, bh))

                digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

                for idx, (bx, by, bw, bh) in enumerate(digit_boxes):
                    digit_img = amount_thresh[by:by+bh, bx:bx+bw]
                    digit_img = cv2.resize(digit_img, (28, 28))
                    digit_img = digit_img.astype("float32") / 255.0
                    digit_img = np.expand_dims(digit_img, axis=-1)
                    digit_img = np.expand_dims(digit_img, axis=0)

                    prediction = model.predict(digit_img, verbose=0)
                    digit = np.argmax(prediction)
                    amount += str(digit)

                found_amount = True
                break

# --- First Name and Last Name Detection ---
results = reader.readtext(image_path)

for i, (bbox, text, prob) in enumerate(results):
    text_upper = text.upper()
    
    # Extract FIRST NAME (accepts both "FIST NAME" and "FIRST NAME")
    if ("FIST" in text_upper or "FIRST" in text_upper) and "NAME" in text_upper:
        next_text = results[i + 1][1] if i + 1 < len(results) else ""
        first_name = re.sub(r"[^a-zA-Z]", "", next_text)  # Keep only letters
    
    # Extract LAST NAME
    elif "LAST" in text_upper and "NAME" in text_upper:
        next_text = results[i + 1][1] if i + 1 < len(results) else ""
        last_name = re.sub(r"[^a-zA-Z]", "", next_text)  # Keep only letters

# --- Save to Database ---
try:
    conn = setup_database()
    
    # Check clients table for correct names if account_number is found
    if found_account and account_number:
        db_first_name, db_last_name = get_client_names(conn, account_number)
        if db_first_name and db_last_name:
            first_name, last_name = db_first_name, db_last_name  # Use names from clients table
        else:
            print("❌ Using extracted names as fallback due to missing client record.")
    
    # Save to customers table if all required attributes are present
    if all([first_name, last_name, account_number, amount]):
        save_to_database(conn, first_name, last_name, account_number, amount)
    else:
        print("❌ Cannot save to database: One or more attributes are missing.")
except Exception as e:
    print(f"❌ Error during database operation: {e}")
finally:
    if 'conn' in locals() and conn.is_connected():
        conn.close()
        print("✅ Database connection closed.")

# --- Final Output ---
print("\n==== FINAL RESULT ====")
if found_account:
    print(f"✅ ACCOUNT NUMBER: {account_number}")
else:
    print("❌ Account number not found.")

if found_amount:
    print(f"✅ AMOUNT: {amount} FRW")
else:
    print("❌ Amount not found.")

if first_name:
    print(f"✅ FIRST NAME: {first_name}")
else:
    print("❌ FIRST NAME not found.")

if last_name:
    print(f"✅ LAST NAME: {last_name}")
else:
    print("❌ LAST NAME not found.")