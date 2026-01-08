import cv2
import numpy as np
import pytesseract
import re
from keras.models import load_model

# Load the model
model = load_model("digit_recognition_cnn_finetuned.h5")

# Load the full slip image
image_path = "../data/raw/3_page-0001.jpg"  # Replace with your path
image = cv2.imread(image_path)

# Preprocess for OCR
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh_ocr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Use Tesseract to find words and their locations
data = pytesseract.image_to_data(thresh_ocr, output_type=pytesseract.Output.DICT)

account_number = ""
found = False

for i, word in enumerate(data['text']):
    if 'account' in word.lower():
        print(f"✅ Found keyword near index {i}")

        # Look for digit-containing box after the word
        for j in range(i + 1, len(data['text'])):
            candidate = data['text'][j]
            if re.search(r'\d', candidate):
                x = data['left'][j]
                y = data['top'][j]
                h = data['height'][j]

                # Manually increase width to get the full account number
                manual_width = 900
                pad = 30
                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + manual_width, image.shape[1])
                y2 = y + h + pad

                # Crop the account number region directly
                cropped = gray[y1:y2, x1:x2]

                # Threshold for digit segmentation
                _, thresh = cv2.threshold(cropped, 150, 255, cv2.THRESH_BINARY_INV)

                # Find contours of digits
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                digit_boxes = []
                for cnt in contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    if bh > 20 and bw > 5:
                        digit_boxes.append((bx, by, bw, bh))

                # Sort boxes left to right
                digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

                # Predict each digit
                for (bx, by, bw, bh) in digit_boxes:
                    digit_img = thresh[by:by+bh, bx:bx+bw]
                    resized = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
                    norm = resized.astype("float32") / 255.0
                    norm = np.expand_dims(norm, axis=-1)
                    norm = np.expand_dims(norm, axis=0)
                    prediction = model.predict(norm, verbose=0)
                    digit_label = np.argmax(prediction)
                    account_number += str(digit_label)

                found = True
                break
        break

# Final output
if found and account_number:
    print(f"✅ ACCOUNT NO: {account_number}")
else:
    print("❌ Account number not detected.")
