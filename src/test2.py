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
    if 'account' in word.lower():  # Detect 'accountno:' or similar
        print(f"✅ Found '{word}' at index {i}")

        # Start from first number-like box after "account"
        for j in range(i + 1, len(data['text'])):
            candidate = data['text'][j]
            if re.search(r'\d', candidate):  # Contains a digit
                x = data['left'][j]
                y = data['top'][j]
                h = data['height'][j]

                # Manually increase the width (force wider crop area)
                manual_width = 900  # You can increase if needed
                pad = 15

                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + manual_width, image.shape[1])
                y2 = y + h + pad

                crop = image[y1:y2, x1:x2]
                cv2.imwrite("account_number_extracted.png", crop)

                print("✅ Forced wide crop saved as 'account_number_extracted.png'")
                found = True
                break
        break

if not found:
    print("❌ Account number field not found.")
