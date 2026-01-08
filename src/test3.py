import cv2
import numpy as np
from keras.models import load_model

# Load your model
model = load_model("digit_recognition_cnn.h5")

# Load the cropped image with account number
img = cv2.imread("account_number_extracted.png", cv2.IMREAD_GRAYSCALE)
_, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours of individual digits
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digit_boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if h > 20 and w > 5:  # Filter noise
        digit_boxes.append((x, y, w, h))

# Sort digits by x-coordinate (left to right)
digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

account_number = ""

for (x, y, w, h) in digit_boxes:
    digit = thresh[y:y+h, x:x+w]
    
    # Resize to 28x28
    resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
    norm = resized.astype("float32") / 255.0
    norm = np.expand_dims(norm, axis=-1)
    norm = np.expand_dims(norm, axis=0)  # Shape: (1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(norm, verbose=0)
    digit_label = np.argmax(prediction)
    account_number += str(digit_label)

# Print final result
print(f"✅ ACCOUNT NO: {account_number}")
