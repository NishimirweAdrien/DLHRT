import cv2
import pytesseract
import re
import numpy as np
from keras.models import load_model

# Load the CNN digit recognition model
model = load_model("digit_recognition_cnn.h5")

# Load the full image
image_path = "../data/raw/slip.jpg"
image = cv2.imread(image_path)

# Preprocess
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# OCR with word-level bounding boxes
data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

found = False
for i, word in enumerate(data['text']):
    if 'amount' in word.lower():  # Detect the keyword 'amount'
        print(f"✅ Found '{word}' at index {i}")

        # Look for digits after 'amount'
        for j in range(i + 1, len(data['text'])):
            candidate = data['text'][j]
            if re.search(r'\d', candidate):  # If it contains a digit
                x = data['left'][j]
                y = data['top'][j]
                h = data['height'][j]

                # Crop a wide area for the amount
                manual_width = 500
                pad = 100

                x1 = max(x - pad + 10, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + manual_width, image.shape[1])
                y2 = y + h + pad - 80

                amount_crop = image[y1:y2, x1:x2]
                gray_amount = cv2.cvtColor(amount_crop, cv2.COLOR_BGR2GRAY)
                _, amount_thresh = cv2.threshold(gray_amount, 150, 255, cv2.THRESH_BINARY_INV)

                # Detect digits using contours
                contours, _ = cv2.findContours(amount_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                digit_boxes = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if h > 20 and w > 5:  # Filter small noise
                        digit_boxes.append((x, y, w, h))
                digit_boxes = sorted(digit_boxes, key=lambda b: b[0])  # Sort left to right

                # Predict each digit
                amount = ""
                for (x, y, w, h) in digit_boxes:
                    digit_img = amount_thresh[y:y+h, x:x+w]
                    digit_img = cv2.resize(digit_img, (28, 28))
                    digit_img = digit_img.astype("float32") / 255.0
                    digit_img = np.expand_dims(digit_img, axis=-1)
                    digit_img = np.expand_dims(digit_img, axis=0)

                    prediction = model.predict(digit_img, verbose=0)
                    digit = np.argmax(prediction)
                    amount += str(digit)

                print(f"✅ AMOUNT DETECTED: {amount} FRW")
                found = True
                break
        break

if not found:
    print("❌ Amount field not found.")
