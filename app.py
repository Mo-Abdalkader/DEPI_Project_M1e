from flask import Flask, request, render_template, jsonify, send_from_directory
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import io
import os

app = Flask(__name__)
model = load_model('gray_Dense.h5')
classes = ['Colon adenocarcinoma', 'Colon benign tissue', 'Lung adenocarcinoma', 'Lung benign tissue', 'Lung squamous cell carcinoma']

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(img):
    if img.mode != "L":  # Convert to grayscale if not already (RGB in case of color)
        img = img.convert("L")
    width, height = img.size
    if width != height:
        delta_w = max(width, height) - width
        delta_h = max(width, height) - height
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        img = ImageOps.expand(img, padding, fill=(0, 0, 0))
    img = img.resize((200, 200))  # Ensure resize method is from PIL.Image
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Expand dimensions for grayscale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(img):
    try:
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        return predicted_class, confidence
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    print(img)
    # Save the image to display it later on the result page
    image_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    img.save(image_filename)

    predictions, confidence = predict_image(img)
    predicted_class = classes[predictions]
    confidence = round(confidence) * 100

    return render_template('index.html', prediction=predicted_class, image_path=image_filename, confidence=confidence)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
