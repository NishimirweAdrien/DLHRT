import easyocr
import cv2
import numpy as np
import pytesseract
import re
from keras.models import load_model

# ========== 1. Initialize Models ==========
# EasyOCR for name extraction
reader = easyocr.Reader(['en'])

# CNN model for digit recognition (account/amount)
model = load_model("digit_recognition_cnn.h5")

# ========== 2. Load Image ==========
image_path = "../data/raw/linda.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# ========== 3. Extract Names (EasyOCR) ==========
results = reader.readtext(image_path)
first_name, last_name = None, None

for i, (bbox, text, prob) in enumerate(results):
    text_upper = text.upper()
    
    # FIRST NAME (handles "FIST" typo)
    if ("FIST" in text_upper or "FIRST" in text_upper) and "NAME" in text_upper:
        next_text = results[i + 1][1] if i + 1 < len(results) else ""
        first_name = re.sub(r"[^a-zA-Z]", "", next_text)
    
    # LAST NAME
    elif "LAST" in text_upper and "NAME" in text_upper:
        next_text = results[i + 1][1] if i + 1 < len(results) else ""
        last_name = re.sub(r"[^a-zA-Z]", "", next_text)

# ========== 4. Extract Account & Amount (Tesseract + CNN) ==========
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh_ocr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
data = pytesseract.image_to_data(thresh_ocr, output_type=pytesseract.Output.DICT)

account_number, amount = "", ""
found_account, found_amount = False, False

for i, word in enumerate(data['text']):
    word_lower = word.lower()

    # --- Account Number ---
    if not found_account and 'account' in word_lower:
        for j in range(i + 1, len(data['text'])):
            if re.search(r'\d', data['text'][j]):
                x, y, h = data['left'][j], data['top'][j], data['height'][j]
                x1 = max(x - 15, 0)
                y1 = max(y - 15, 0)
                x2 = min(x + 900, image.shape[1])
                y2 = y + h + 15

                cropped = gray[y1:y2, x1:x2]
                _, thresh_account = cv2.threshold(cropped, 150, 255, cv2.THRESH_BINARY_INV)

                contours, _ = cv2.findContours(thresh_account, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                digit_boxes = sorted(
                    [cv2.boundingRect(cnt) for cnt in contours if cv2.boundingRect(cnt)[2] > 5 and cv2.boundingRect(cnt)[3] > 20],
                    key=lambda b: b[0]
                )

                for (bx, by, bw, bh) in digit_boxes:
                    digit_img = cv2.resize(thresh_account[by:by+bh, bx:bx+bw], (28, 28))
                    digit_img = np.expand_dims(digit_img.astype("float32") / 255.0, axis=(0, -1))
                    account_number += str(np.argmax(model.predict(digit_img, verbose=0)))

                found_account = True
                break

    # --- Amount ---
    if not found_amount and 'amount' in word_lower:
        for j in range(i + 1, len(data['text'])):
            if re.search(r'\d', data['text'][j]):
                x, y, h = data['left'][j], data['top'][j], data['height'][j]
                x1 = max(x - 90, 0)
                y1 = max(y - 100, 0)
                x2 = min(x + 500, image.shape[1])
                y2 = y + h + 20

                amount_crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                _, amount_thresh = cv2.threshold(amount_crop, 150, 255, cv2.THRESH_BINARY_INV)

                contours, _ = cv2.findContours(amount_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                digit_boxes = sorted(
                    [cv2.boundingRect(cnt) for cnt in contours if cv2.boundingRect(cnt)[2] > 5 and cv2.boundingRect(cnt)[3] > 20],
                    key=lambda b: b[0]
                )

                for (bx, by, bw, bh) in digit_boxes:
                    digit_img = cv2.resize(amount_thresh[by:by+bh, bx:bx+bw], (28, 28))
                    digit_img = np.expand_dims(digit_img.astype("float32") / 255.0, axis=(0, -1))
                    amount += str(np.argmax(model.predict(digit_img, verbose=0)))

                found_amount = True
                break

# ========== 5. Print Results ==========
print("\n==== FINAL RESULT ====")
print(f"FIRST NAME: {first_name if first_name else 'Not found'}")
print(f"LAST NAME: {last_name if last_name else 'Not found'}")
print(f"ACCOUNT NUMBER: {account_number if found_account else 'Not found'}")
print(f"AMOUNT: {amount if found_amount else 'Not found'} FRW")