import os
import sqlite3
import json
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import random

# Load MobileNetV2
mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Configuration
image_directory = 'static/images/'  # Path to the folder containing the images
db_path = 'database/products.db'

# Function to extract features
def extract_features_with_mobilenet(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to load image: {img_path}")
    
    img = cv2.resize(img, (224, 224))  # Resize for MobileNetV2
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = mobilenet_model.predict(img_array)
    return features.flatten().tolist()

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Browse image files
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

for image_file in image_files:
    name = os.path.splitext(image_file)[0]
    price = round(random.uniform(10, 200), 2)
    image_url = f"/{image_directory}{image_file}"
    img_path = os.path.join(image_directory, image_file)

    try:
        feature_vector = extract_features_with_mobilenet(img_path)
        feature_vector_json = json.dumps(feature_vector)  # Convert to JSON
        
        # Insert into the database
        cursor.execute('''
            INSERT INTO products (name, price, image_url, feature_vector) 
            VALUES (?, ?, ?, ?)
        ''', (name, price, image_url, feature_vector_json))
    except Exception as e:
        print(f"Error with image {image_file}: {e}")

# Save changes and close the connection
conn.commit()
conn.close()

print("Images processed and successfully added to the database.")
