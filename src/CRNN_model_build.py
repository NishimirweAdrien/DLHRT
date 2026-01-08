import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# 📥 Load your EMNIST CSV file
print("📥 Loading EMNIST CSV...")
df = pd.read_csv("../emnist/emnist-byclass-train.csv")  # Adjust to the correct filename

# 🏷 Separate labels and images
labels = df.iloc[:, 0].values
images = df.iloc[:, 1:].values

# 🖼 Normalize and reshape
images = images.astype('float32') / 255.0
images = images.reshape((-1, 28, 28, 1))  # Add channel dimension

# 🔢 One-hot encode labels
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes)

# 🧪 Split into train/test
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=42)

# ✅ Your data is now ready
print(f"✅ Dataset loaded: {x_train.shape[0]} training samples, {x_val.shape[0]} validation samples")
