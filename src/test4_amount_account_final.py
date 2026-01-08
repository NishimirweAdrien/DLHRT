import cv2
import numpy as np
import pytesseract
import re
from keras.models import load_model

# Load the CNN digit recognition model
model = load_model("digit_recognition_cnn.h5")

# Load the full slip image
image_path = "../data/raw/3_page-0001.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh_ocr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# OCR to extract word locations
data = pytesseract.image_to_data(thresh_ocr, output_type=pytesseract.Output.DICT)

account_number = ""
amount = ""
found_account = False
found_amount = False

for i, word in enumerate(data['text']):
    word_lower = word.lower()

    # ---------- Account Number Detection ----------
    if not found_account and 'account' in word_lower:
        print(f"✅ Found 'account' at index {i}")
        for j in range(i + 1, len(data['text'])):
            if re.search(r'\d', data['text'][j]):
                x = data['left'][j]
                y = data['top'][j]
                h = data['height'][j]
                manual_width = 900
                pad = 30

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
                    if bw > 10 and bh > 25:  # Stricter size filter
                        digit_boxes.append((bx, by, bw, bh))

                digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

                # Limit to expected number of digits (9 for account number)
                digit_boxes = digit_boxes[:9]

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

    # ---------- Amount Detection ----------
    if not found_amount and 'amount' in word_lower:
        print(f"✅ Found 'amount' at index {i}")
        for j in range(i + 1, len(data['text'])):
            if re.search(r'\d', data['text'][j]):
                x = data['left'][j]
                y = data['top'][j]
                h = data['height'][j]
                manual_width = 380
                pad = 10

                x1 = max(x - 50, 0)
                y1 = max(y - 80, 0)
                x2 = min(x + manual_width, image.shape[1])
                y2 = y + h + pad

                amount_crop = image[y1:y2, x1:x2]
                gray_amount = cv2.cvtColor(amount_crop, cv2.COLOR_BGR2GRAY)
                _, amount_thresh = cv2.threshold(gray_amount, 150, 255, cv2.THRESH_BINARY_INV)

                contours, _ = cv2.findContours(amount_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                digit_boxes = []
                for cnt in contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    if bw > 10 and bh > 25:  # Stricter size filter
                        digit_boxes.append((bx, by, bw, bh))

                digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

                # Limit to expected number of digits (7 for amount)
                digit_boxes = digit_boxes[:7]

                for (bx, by, bw, bh) in digit_boxes:
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

# Final output
print("\n==== FINAL RESULT ====")
if found_account:
    print(f"✅ ACCOUNT NUMBER: {account_number}")
else:
    print("❌ Account number not found.")

if found_amount:
    print(f"✅ AMOUNT: {amount} FRW")
else:
    print("❌ Amount not found.")