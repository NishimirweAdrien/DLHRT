import cv2
import numpy as np
import os
import re
import pytesseract
from keras.models import load_model
import time

# Load the CNN digit recognition model
model = load_model("digit_recognition_cnn.h5")

# Load the full slip image
image_path = "../data/raw/slip.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh_account = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# OCR to extract word locations
data = pytesseract.image_to_data(thresh_account, output_type=pytesseract.Output.DICT)

for i, word in enumerate(data['text']):
    word_lower = word.lower()
    if 'account' in word_lower:
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
                    if bw > 10 and bh > 25:
                        digit_boxes.append((bx, by, bw, bh))

                digit_boxes = sorted(digit_boxes, key=lambda b: b[0])
                digit_boxes = digit_boxes[:9]  # Limit to 9 digits

                for idx, (bx, by, bw, bh) in enumerate(digit_boxes):
                    digit_img = thresh_account[by:by+bh, bx:bx+bw]
                    digit_img = cv2.resize(digit_img, (28, 28))
                    digit_img = digit_img.astype("float32") / 255.0
                    digit_img = np.expand_dims(digit_img, axis=-1)
                    digit_img = np.expand_dims(digit_img, axis=0)

                    prediction = model.predict(digit_img, verbose=0)
                    digit = np.argmax(prediction)
                    print(f"Digit {idx}: Predicted {digit}, Saving image...")
                    if not os.path.exists(f"dataset/{digit}/"):
                        os.makedirs(f"dataset/{digit}/")
                    # Save the original cropped image before resizing
                    save_img = np.uint8(thresh_account[by:by+bh, bx:bx+bw])
                    if save_img.size > 0 and save_img.max() > 0:  # Validate image
                        print(f"Image shape: {save_img.shape}, min: {save_img.min()}, max: {save_img.max()}")
                        unique_id = int(time.time() * 1000)  # Unique timestamp
                        cv2.imwrite(f"dataset/{digit}/digit_{idx}_{unique_id}.png", save_img)
                    else:
                        print("Skipping empty or invalid image.")

                break