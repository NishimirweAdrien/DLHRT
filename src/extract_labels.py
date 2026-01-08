import cv2
import pytesseract
from difflib import SequenceMatcher
import os
import tensorflow as tf

# Function to preprocess the image
def preprocess_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return binary

# Function to match detected text with expected labels
def match_label(text, labels, threshold=0.7):
    text = text.upper().strip()
    for label in labels:
        similarity = SequenceMatcher(None, text, label).ratio()
        if similarity >= threshold:
            return label
    return None

# Main function to extract labels
def extract_labels(image_path, model_path):
    try:
        print("Loading digit recognition model...")
        digit_model = tf.keras.models.load_model(model_path)
        print("Digit recognition model loaded successfully.")

        print("Loading image...")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        print("Image loaded successfully.")
        print(f"Image dimensions: {img.shape}")

        print("Preprocessing image...")
        img_processed = preprocess_image(img)
        debug_dir = "../data/debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        cv2.imwrite(os.path.join(debug_dir, "preprocessed_binary.png"), img_processed)
        print(f"Saved preprocessed image to {os.path.join(debug_dir, 'preprocessed_binary.png')}")

        # Define expected labels and their approximate bounding boxes
        label_bboxes = {
            "first_name": {
                "labels": ["FIRST NAME:", "FIST NAME:", "FST NAME", "FIRSTNAME", "F NAME", "FNAME"],
                "bbox": [[50, 50], [150, 50], [150, 80], [50, 80]]
            },
            "last_name": {
                "labels": ["LAST NAME:", "LASTNAME", "LST NAME", "L NAME", "LNAME"],
                "bbox": [[50, 100], [150, 100], [150, 130], [50, 130]]
            },
            "account_no": {
                "labels": ["ACCOUNT NO:", "ACCOUNTNO", "ACC NO", "ACCNO", "ACCT NO", "ACCTNO"],
                "bbox": [[50, 150], [150, 150], [150, 180], [50, 180]]
            },
            "amount": {
                "labels": ["AMOUNT:", "AMOUNT", "AMT:", "AMT"],
                "bbox": [[50, 200], [150, 200], [150, 230], [50, 230]]
            },
            "currency": {
                "labels": ["FRW", "USD", "FRI", "RW", "US"],
                "bbox": [[150, 200], [200, 200], [200, 230], [150, 230]]
            }
        }

        detected_labels = {
            "first_name": None,
            "last_name": None,
            "account_no": None,
            "amount": None,
            "currency": None
        }

        # Process each label region
        print("Processing label regions...")
        for field, info in label_bboxes.items():
            labels = info["labels"]
            bbox = info["bbox"]
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            
            cropped_img = img_processed[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(debug_dir, f"cropped_{field}.png"), cropped_img)
            print(f"Saved cropped region for {field} to {os.path.join(debug_dir, f'cropped_{field}.png')}")

            config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ: -c preserve_interword_spaces=1'
            results = pytesseract.image_to_data(cropped_img, lang='eng', output_type=pytesseract.Output.DICT, config=config)
            
            print(f"\nDetected text in {field} region:")
            found_label = False
            for i in range(len(results['text'])):
                text = results['text'][i].strip()
                if text:
                    confidence = float(results['conf'][i]) / 100.0
                    print(f"Text: {text}, Confidence: {confidence:.2f}")
                    matched_label = match_label(text, labels, threshold=0.7)
                    if matched_label:
                        detected_labels[field] = (matched_label, bbox)
                        print(f"Matched {field} label: {matched_label}, BBox: {bbox}")
                        found_label = True
                        break
            
            if not found_label:
                print(f"No match found for {field}, assuming label is present at BBox: {bbox}")
                detected_labels[field] = (labels[0], bbox)

        print("\nExtracted labels:")
        for field, label_info in detected_labels.items():
            if label_info:
                label, bbox = label_info
                print(f"{field}: {label}, BBox: {bbox}")
            else:
                print(f"{field}: Not detected")

        return detected_labels

    except Exception as e:
        print(f"Error in extract_labels: {str(e)}")
        raise

if __name__ == "__main__":
    image_path = "../data/raw/slip.jpg"
    model_path = "../models/digit_cnn_best.h5"
    detected_labels = extract_labels(image_path, model_path)