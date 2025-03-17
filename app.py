import os
import cv2
import numpy as np
import json
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import sqlite3
import faiss
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Flask app configuration
app = Flask(__name__)
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

mobilenet_v2_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Function to check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to connect to the database
def connect_db():
    return sqlite3.connect('database/products.db')

def extract_features_with_mobilenet_v2(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Unable to load image: {img_path}")
    
    img = cv2.resize(img, (224, 224))  # Required size for MobileNetV2
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # MobileNetV2 specific preprocessing
    
    features = mobilenet_v2_model.predict(img_array)  # Extract features
    return features.flatten()

# Function to create a FAISS index
def create_faiss_index(feature_vectors):
    d = feature_vectors.shape[1]
    index = faiss.IndexFlatL2(d)  # Index based on Euclidean distance
    index.add(feature_vectors.astype('float32'))  # Add the vectors
    return index

# Retrieve feature vectors and image paths from the database
def get_all_images_from_db():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, price, image_url, feature_vector FROM products")
    images = cursor.fetchall()
    conn.close()

    feature_vectors = []
    image_paths = []

    for image_id, name, price, img_url, feature_vector_json in images:
        features = np.array(json.loads(feature_vector_json))
        feature_vectors.append(features)
        image_paths.append((image_id, img_url, name, price))
    
    return np.array(feature_vectors), image_paths

# Find similar images using FAISS
def find_similar_images(query_image_path, db_feature_vectors, db_image_paths, top_n=5):
    query_features = extract_features_with_mobilenet_v2(query_image_path)
    query_features = query_features.reshape(1, -1).astype('float32')  # FAISS expects float32

    index = create_faiss_index(db_feature_vectors)
    D, I = index.search(query_features, top_n)

    similar_images = [(db_image_paths[i], D[0][j]) for j, i in enumerate(I[0])]
    return similar_images

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, price, image_url FROM products LIMIT 10")
    default_images = cursor.fetchall()
    conn.close()

    default_images_list = [{
        'id': img[0],
        'path': img[3],
        'name': img[1],
        'price': img[2]
    } for img in default_images]

    similar_images = []
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            db_feature_vectors, db_image_paths = get_all_images_from_db()
            if db_feature_vectors.size == 0:
                return render_template('index.html', default_images=default_images_list, similar_images=None, error="No image found.")

            similar_images = find_similar_images(filepath, db_feature_vectors, db_image_paths)
            return render_template('index.html', default_images=default_images_list, similar_images=similar_images, error=None)

    return render_template('index.html', default_images=default_images_list, similar_images=None, error=None)

def get_product_by_id(product_id):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, price, image_url FROM products WHERE id = ?", (product_id,))
    product = cursor.fetchone()
    conn.close()
    if product:
        return {
            'id': product[0],
            'name': product[1],
            'price': product[2],
            'image_url': product[3]
        }
    return None
    
@app.route('/add_to_cart/<int:product_id>', methods=['POST'])
def add_to_cart(product_id):
    product = get_product_by_id(product_id)
    if not product:
        return redirect(url_for('index'))
    
    # Add the product to the cart in the session
    if 'cart' not in session:
        session['cart'] = []

    session['cart'].append(product)
    session.modified = True  # Indicate that the session has been modified
    
    return redirect(url_for('index'))

# Route to display the cart
@app.route('/cart')
def cart():
    cart_items = session.get('cart', [])
    total = sum(float(item['price']) for item in cart_items)
    return render_template('cart.html', cart_items=cart_items, total=total)

@app.route('/view_cart')
def view_cart():
    cart = session.get('cart', [])  # Retrieve the contents of the cart, if any
    return render_template('cart.html', cart=cart)  # Display the cart page with the products

@app.route('/clear_cart')
def clear_cart():
    session.pop('cart', None)  # Remove the cart from the session
    return redirect(url_for('view_cart'))  # Redirect to the cart page

if __name__ == '__main__':
    app.run(debug=True)
