import cv2
import numpy as np
import os

def preprocess_image(image_path, output_path, target_height=512, target_width=4096):
    """
    Preprocess a handwritten slip image for CRNN input.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        target_height (int): Desired height for the output image.
        target_width (int): Desired width for the output image.
    
    Returns:
        np.array: Preprocessed image ready for CRNN, or None if failed.
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not open or find the image: {image_path}")
            return None

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Converted to grayscale: {image_path}, shape={gray_image.shape}")

        # Crop the region of interest (ROI) - use full width to capture all text
        h, w = gray_image.shape
        roi_x, roi_y, roi_w, roi_h = 0, 0, w, h  # Full image width
        roi_image = gray_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        print(f"Cropped ROI: shape=({roi_h}, {roi_w})")

        # Resize while maintaining aspect ratio
        h, w = roi_image.shape
        aspect_ratio = w / h
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
        resized_image = cv2.resize(roi_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        print(f"Resized image: shape=({new_height}, {new_width})")

        # Pad or crop to target width
        if new_width != target_width:
            if new_width < target_width:
                # Pad with white pixels (255) to match typical CRNN input
                pad_width = target_width - new_width
                resized_image = cv2.copyMakeBorder(resized_image, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=255)
            else:
                # Crop to target width
                resized_image = resized_image[:, :target_width]
        print(f"Final image shape: ({target_height}, {target_width})")

        # Normalize pixel values to [0, 1] for CRNN
        normalized_image = resized_image.astype(np.float32) / 255.0
        print(f"Normalized image: min={normalized_image.min()}, max={normalized_image.max()}")

        # Add channel dimension for CRNN (height, width, channels)
        processed_image = np.expand_dims(normalized_image, axis=-1)

        # Save only the processed image
        image_to_save = (normalized_image * 255).astype(np.uint8)
        cv2.imwrite(output_path, image_to_save)
        print(f"Saved processed image: {output_path}")

        return processed_image

    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def preprocess_dataset(input_dir, output_dir, target_height=512, target_width=4096):
    """
    Preprocess all images in the input directory and save to output directory.
    
    Args:
        input_dir (str): Directory containing raw images.
        output_dir (str): Directory to save preprocessed images.
        target_height (int): Desired height for output images.
        target_width (int): Desired width for output images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"processed_{filename}")
            
            # Preprocess the image
            processed_image = preprocess_image(input_path, output_path, target_height, target_width)
            if processed_image is None:
                print(f"Failed to process: {input_path}")

if __name__ == "__main__":
    # Example usage
    input_dir = "../../data/raw"
    output_dir = "../../data/processed"
    preprocess_dataset(input_dir, output_dir)
