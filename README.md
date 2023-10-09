- 👋 Hi, I’m @ABINAYA155360
- 👀 I’m interested in learn new things.
- 🌱 I’m currently learning data analysis
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ..my mail id is abinayabi55@gmail.com

<!---
ABINAYA155360/ABINAYA155360 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Load and preprocess data (adjust based on the dataset structure)
# Assume you have a directory structure with 'bleeding' and 'non_bleeding' folders

def load_data(directory):
    data = []
    labels = []
    
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            
            # Preprocess the image (resize, normalize, etc.)
            img = preprocess_image(img_path)
            
            data.append(img)
            labels.append(1 if category == 'bleeding' else 0)  # 1 for bleeding, 0 for non-bleeding
    
    return np.array(data), np.array(labels)

# Define a simple convolutional neural network (CNN) model
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification (bleeding or non-bleeding)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the data
data, labels = load_data('/path/to/dataset')

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create the model
input_shape = data[0].shape
model = create_model(input_shape)

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))
