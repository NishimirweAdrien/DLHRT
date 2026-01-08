import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_for_ocr(img):
    """Enhanced preprocessing specifically for form fields"""
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remove small noise
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return Image.fromarray(cleaned)

def find_field_region(image_path, field_name):
    """More robust field detection using contour analysis"""
    # Load original image
    original = cv2.imread(image_path)
    img = preprocess_for_ocr(Image.open(image_path))
    
    # Get OCR data with bounding boxes
    d = pytesseract.image_to_data(np.array(img), output_type=pytesseract.Output.DICT,
                                config='--psm 6')
    
    # Find all instances of the field name
    matches = []
    for i, text in enumerate(d['text']):
        if field_name.lower() in text.lower():
            matches.append({
                'text': text,
                'left': d['left'][i],
                'top': d['top'][i],
                'width': d['width'][i],
                'height': d['height'][i]
            })
    
    if not matches:
        print(f"'{field_name}' not found in OCR results. Trying alternative methods...")
        return find_field_by_contours(original, field_name)
    
    # Get the best match (largest bounding box)
    best_match = max(matches, key=lambda x: x['width']*x['height'])
    
    # Expand region to include value (right side)
    padding = 20
    x1 = best_match['left'] - padding
    y1 = best_match['top'] - padding
    x2 = x1 + best_match['width'] + 300  # Extra space for value
    y2 = y1 + best_match['height'] + 2*padding
    
    # Ensure coordinates are within image bounds
    h, w = original.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    return (x1, y1, x2, y2)

def find_field_by_contours(img, field_name):
    """Fallback method using contour detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        roi = img[y:y+h, x:x+w]
        
        # OCR just this region
        text = pytesseract.image_to_string(roi, config='--psm 6')
        if field_name.lower() in text.lower():
            return (x, y, x+w+200, y+h)  # Expand right for value
    
    return None

def crop_and_save_field(image_path, field_name, output_path):
    """Main function to crop and save field"""
    region = find_field_region(image_path, field_name)
    
    if region:
        img = Image.open(image_path)
        cropped = img.crop(region)
        cropped.save(output_path)
        print(f"✅ Successfully saved {field_name} region to {output_path}")
        return True
    else:
        print(f"❌ Failed to locate {field_name} region")
        return False

if __name__ == "__main__":
    # Example usage
    crop_and_save_field(
        image_path="../../data/raw/slip.jpg",
        field_name="ACCOUNT NO",
        output_path="account_no_region.png"
    )