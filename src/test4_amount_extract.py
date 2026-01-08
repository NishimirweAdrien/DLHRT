import cv2
import pytesseract
import re

# Load the image
image_path = "../data/raw/slip.jpg"  # Replace with your image file
image = cv2.imread(image_path)

# Preprocess
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# OCR with box data
data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

found = False
for i, word in enumerate(data['text']):
    if 'amount' in word.lower():  # Detect the keyword 'amount'
        print(f"✅ Found '{word}' at index {i}")

        # Look for digits after the keyword
        for j in range(i + 1, len(data['text'])):
            candidate = data['text'][j]
            if re.search(r'\d', candidate):  # If it contains a digit
                x = data['left'][j]
                y = data['top'][j]
                h = data['height'][j]

                # Crop a wide area for the number
                manual_width = 360  # Increase if numbers are long
                pad = 10

                x1 = max(x - 50, 0)
                y1 = max(y - 80,0)
                x2 = min(x + manual_width, image.shape[1])
                y2 = y + h + pad

                crop = image[y1:y2, x1:x2]
                cv2.imwrite("amount_extracted.png", crop)

                print("✅ Amount area saved as 'amount_extracted.png'")
                found = True
                break
        break

if not found:
    print("❌ Amount field not found.")
