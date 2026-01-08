
import os
import cv2
import numpy as np
import albumentations as A

def augment_digit_images(input_dir, output_dir, augmentations_per_image=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])

    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            digit_label = filename.split('_')[1]
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            # Ensure the image is 28x28
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0

            for i in range(augmentations_per_image):
                augmented = transform(image=img)
                aug_img = augmented['image']
                aug_img = (aug_img * 255).astype(np.uint8)
                # Ensure the output is 28x28
                aug_img = cv2.resize(aug_img, (28, 28), interpolation=cv2.INTER_AREA)
                aug_filename = f"aug_digit_{digit_label}_{i}.png"
                aug_filepath = os.path.join(output_dir, aug_filename)
                cv2.imwrite(aug_filepath, aug_img)

if __name__ == "__main__":
    input_dir = "../data/digits"
    output_dir = "../data/augmented_digits"
    augment_digit_images(input_dir, output_dir, augmentations_per_image=50)
    print(f"Augmented digit images saved to {output_dir}")
