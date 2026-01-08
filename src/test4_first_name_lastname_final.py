import easyocr
import re

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Image path
image_path = '../data/raw/slip.jpg'

# Run OCR
results = reader.readtext(image_path)

# Initialize variables
first_name = None
last_name = None

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

# Print results
if first_name:
    print(f"FIRST NAME: {first_name}")
else:
    print("FIRST NAME not found.")

if last_name:
    print(f"LAST NAME: {last_name}")
else:
    print("LAST NAME not found.")