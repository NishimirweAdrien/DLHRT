import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import os
import matplotlib.pyplot as plt

# Ensure the output directory exists
output_dir = '../data/processed/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# 1. Load the saved model
try:
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure 'mnist_cnn_model.h5' exists in the same directory.")
    exit(1)

# 2. Function to preprocess and predict a sequence of digits from a custom image
def predict_digit_sequence(image_path, true_sequence, invert_colors=True, enhance_contrast=True):
    try:
        # Load the image with PIL
        image = Image.open(image_path)
        print(f"Image format: {image.format}")
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        img_width, img_height = image.size

        # Convert to grayscale
        img = image.convert("L")
        img.save(os.path.join(output_dir, 'debug_grayscale.png'))
        print("Grayscale image saved as 'debug_grayscale.png'")

        # Enhance contrast if specified
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)  # Adjustable contrast
            img.save(os.path.join(output_dir, 'debug_contrast.png'))
            print("Contrast-enhanced image saved as 'debug_contrast.png'")

        # Apply sharpening filter
        img = img.filter(ImageFilter.SHARPEN)

        # Convert to numpy array for OpenCV processing
        img_array = np.array(img)

        # Print grayscale statistics
        print(f"Grayscale stats - Min: {img_array.min()}, Max: {img_array.max()}, Mean: {img_array.mean():.2f}")

        # Apply adaptive thresholding to binarize the image
        img_array = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 2)  # Reduced blockSize and C
        cv2.imwrite(os.path.join(output_dir, 'debug_threshold.png'), img_array)
        print("Thresholded image saved as 'debug_threshold.png'")

        # Apply morphological operations to clean up and separate digits
        kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
        img_array = cv2.erode(img_array, kernel, iterations=1)  # Remove small noise
        img_array = cv2.dilate(img_array, kernel, iterations=1)  # Thicken digits, reduced to 1 iteration
        cv2.imwrite(os.path.join(output_dir, 'debug_morphology.png'), img_array)
        print("Image after morphological operations saved as 'debug_morphology.png'")

        # Find contours to detect all digits
        contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} raw contours before filtering.")

        # Visualize all raw contours
        img_with_raw_contours = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_with_raw_contours, (x, y), (x + w, y + h), (0, 0, 255), 1)  # Red boxes
            cv2.putText(img_with_raw_contours, f"C{idx}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(output_dir, 'debug_raw_contours.png'), img_with_raw_contours)
        print("Image with raw contours saved as 'debug_raw_contours.png'")

        if not contours:
            print("Warning: No contours found. Unable to detect digits.")
            return None

        # Filter contours by area and size to avoid noise
        max_area = img_width * img_height * 0.3  # Max area is 30% of image size
        contours = [c for c in contours if 100 < cv2.contourArea(c) < max_area]  # Lowered min area to 100
        contours = [c for c in contours if cv2.boundingRect(c)[2] > 10 and cv2.boundingRect(c)[3] > 10]  # Lowered to 10
        if not contours:
            print("Warning: No valid contours found after filtering.")
            print("Contour areas:", [cv2.contourArea(c) for c in contours])
            return None

        # Print contour details for debugging
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            print(f"Contour {idx}: Area={area:.2f}, Bounding box: x={x}, y={y}, w={w}, h={h}")

        # Sort contours by x-coordinate (left to right)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        print(f"Detected {len(contours)} digits in the image.")

        # Force splitting to match the true number of digits (11)
        expected_digits = len(true_sequence)
        print(f"Expected {expected_digits} digits based on true sequence.")
        if len(contours) != expected_digits:
            print(f"Detected {len(contours)} digits, but expected {expected_digits}. Forcing split...")
            new_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                print(f"Contour bounding box: x={x}, y={y}, w={w}, h={h}")
                est_num_digits = max(1, int(w / (img_width / expected_digits)))
                if est_num_digits > 1:
                    segment_width = w // est_num_digits
                    for i in range(est_num_digits):
                        new_x = x + i * segment_width
                        new_contour = np.array([[[new_x, y]], [[new_x + segment_width, y]],
                                               [[new_x + segment_width, y + h]], [[new_x, y + h]]])
                        new_contours.append(new_contour)
                else:
                    new_contours.append(contour)
            contours = sorted(new_contours, key=lambda c: cv2.boundingRect(c)[0])
            print(f"After splitting, detected {len(contours)} digits.")

        # If still more contours than expected, take the largest ones
        if len(contours) > expected_digits:
            contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)[:expected_digits]
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
            print(f"Reduced to {len(contours)} digits by selecting largest contours.")

        # Visualize contours on the original image
        img_with_contours = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes
            cv2.putText(img_with_contours, f"Digit {idx}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, 'debug_contours.png'), img_with_contours)
        print("Image with detected contours saved as 'debug_contours.png'")

        # Process each digit and predict
        predicted_sequence = []
        preprocessed_digits = []
        for idx, contour in enumerate(contours):
            # Get bounding box for the digit
            x, y, w, h = cv2.boundingRect(contour)
            print(f"Digit {idx} bounding box: x={x}, y={y}, w={w}, h={h}")

            # Crop the digit with some padding
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2 * padding)
            h = min(img_array.shape[0] - y, h + 2 * padding)
            digit_crop = img_array[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, f'debug_cropped_digit_{idx}.png'), digit_crop)
            print(f"Cropped digit {idx} saved as 'debug_cropped_digit_{idx}.png'")

            # Resize while preserving aspect ratio
            aspect_ratio = w / h
            target_size = 20
            if aspect_ratio > 1:  # Wider than tall
                new_w = target_size
                new_h = int(target_size / aspect_ratio)
            else:  # Taller than wide or square
                new_h = target_size
                new_w = int(target_size * aspect_ratio)
            new_w = max(10, new_w)
            new_h = max(10, new_h)
            digit_resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(output_dir, f'debug_resized_digit_{idx}.png'), digit_resized)
            print(f"Resized digit {idx} to ({new_w}, {new_h}) saved as 'debug_resized_digit_{idx}.png'")

            # Create a 28x28 canvas and center the digit
            canvas = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
            digit_array = canvas

            # Invert colors if specified (MNIST: white digits on black background)
            if invert_colors:
                digit_array = 255 - digit_array

            # Normalize pixel values
            digit_array = digit_array.astype('float32') / 255.0

            # Save preprocessed digit
            cv2.imwrite(os.path.join(output_dir, f'preprocessed_digit_{idx}.png'), digit_array * 255)
            print(f"Preprocessed digit {idx} saved as 'preprocessed_digit_{idx}.png'")
            preprocessed_digits.append(digit_array * 255)

            # Reshape for model input
            digit_array = digit_array.reshape(1, 28, 28, 1)

            # Predict the digit
            prediction = model.predict(digit_array)
            predicted_digit = np.argmax(prediction[0])
            probabilities = prediction[0]

            # Print prediction probabilities for this digit
            print(f"\nPrediction probabilities for digit {idx} (0-9):")
            for i, prob in enumerate(probabilities):
                print(f"Digit {i}: {prob:.4f}")
            print(f"Predicted digit {idx}: {predicted_digit}")

            predicted_sequence.append(str(predicted_digit))

        # Combine the predicted digits into a sequence
        predicted_sequence_str = ''.join(predicted_sequence)

        # Visualize preprocessed digits with predicted and true labels
        num_digits = len(predicted_sequence)
        true_digits = list(true_sequence)
        fig, axes = plt.subplots(2, num_digits, figsize=(num_digits * 2, 4))
        for idx in range(num_digits):
            # Plot preprocessed digit
            axes[0, idx].imshow(preprocessed_digits[idx], cmap='gray')
            axes[0, idx].set_title(f"Pred: {predicted_sequence[idx]}")
            axes[0, idx].axis('off')
            # Plot true digit (using MNIST sample for visualization)
            if idx < len(true_digits):
                (x_test, y_test) = tf.keras.datasets.mnist.load_data()[1]
                true_digit = int(true_digits[idx])
                sample_idx = np.where(y_test == true_digit)[0][0]
                sample_digit = x_test[sample_idx]
                axes[1, idx].imshow(sample_digit, cmap='gray')
                axes[1, idx].set_title(f"True: {true_digit}")
                axes[1, idx].axis('off')
            else:
                axes[1, idx].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_comparison.png'))
        plt.close()
        print("Prediction comparison plot saved as 'prediction_comparison.png'")

        return predicted_sequence_str
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# 3. Test model on a few MNIST test images to verify behavior
def test_mnist_images():
    print("\nTesting model on MNIST test images...")
    (x_test, y_test) = tf.keras.datasets.mnist.load_data()[1]
    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    for i in range(3):  # Test 3 random images
        img = x_test[i:i+1]
        true_digit = y_test[i]
        pred = model.predict(img)
        pred_digit = np.argmax(pred[0])
        print(f"MNIST test image {i+1}: True digit: {true_digit}, Predicted: {pred_digit}")

# 4. Example usage for custom image prediction
image_path = '../data/raw/sequence.jpg'  # Path to your image
true_sequence = "40011132029"  # True sequence for comparison
invert_colors = True  # Set to False if your digits are white on black
enhance_contrast = True  # Set to False to disable contrast enhancement

# Test MNIST images first
test_mnist_images()

# Predict on custom image with a sequence of digits
if os.path.exists(image_path):
    predicted_sequence = predict_digit_sequence(image_path, true_sequence, invert_colors=invert_colors,
                                               enhance_contrast=enhance_contrast)
    if predicted_sequence is not None:
        print(f"\nThe predicted digit sequence is: {predicted_sequence}")
else:
    print(f"Image not found: {image_path}. Please check the path.")