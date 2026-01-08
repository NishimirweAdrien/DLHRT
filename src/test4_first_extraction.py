import cv2
import pytesseract
import re

# Load the image
image_path = "../data/raw/slip.jpg"
image = cv2.imread(image_path)

# Preprocess
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# OCR with word-level bounding boxes
data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

found = False
for i, word in enumerate(data['text']):
    if 'name' in word.lower():  # Detect keyword 'name' or 'first name'
        print(f"✅ Found keyword '{word}' at index {i}")
        
        for j in range(i + 1, len(data['text'])):
            candidate = data['text'][j].strip()
            if candidate and re.match(r'^[A-Za-z]+$', candidate):  # Likely a name
                print(f"✅ FIRST NAME DETECTED: {candidate}")
                
                # Crop and save the name region using custom width
                x = data['left'][j]
                y = data['top'][j]
                h = data['height'][j]

                manual_width = 600  # Customize this width
                manual_height = 120  # Optional: customize height
                pad = 10  # Extra padding if needed

                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + manual_width, image.shape[1])
                y2 = min(y + manual_height, image.shape[0])

                name_crop = image[y1:y2, x1:x2]
                cv2.imwrite("first_name_crop.jpg", name_crop)
                print("📸 Cropped image saved as 'first_name_crop.jpg'")

                found = True
                break
        break

if not found:
    print("❌ First name field not found.")
