from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops, ImageEnhance
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image based on the selected model
def preprocess_image(img_path, model_name):
    img = image.load_img(img_path, target_size=(128, 128))
    
    if model_name == 'xception':
        img_array = image.img_to_array(img)
        img_array = preprocess_input_xception(np.expand_dims(img_array, axis=0))
    elif model_name == 'densenet':
        img_array = image.img_to_array(img)
        img_array = preprocess_input_densenet(np.expand_dims(img_array, axis=0))
    elif model_name == 'vgg16':
        img_array = image.img_to_array(img)
        img_array = preprocess_input_vgg16(np.expand_dims(img_array, axis=0))
    else:
        raise ValueError("Invalid model name")
    
    return img_array

# Function to make predictions
def make_predictions(img_array, model):
    predictions = model.predict(img_array)
    return predictions

# Function to perform ELA and make predictions
def apply_ela_and_predict(input_path, model):
    for filename in os.listdir(input_path):
        img_path = os.path.join(input_path, filename)

        if img_path.lower().endswith(('.jpg', '.jpeg')):
            # Open the image and convert it to RGB mode
            image = Image.open(img_path).convert('RGB')
            
            # Resave the image with the specified quality
            image.save('temp.jpg', 'JPEG', quality=90)
            resaved = Image.open('temp.jpg')

            # Calculate the ELA (Error Level Analysis) image by taking the difference between the original and resaved image
            ela_image = ImageChops.difference(image, resaved)

            # Get the minimum and maximum pixel values in the ELA image
            band_values = ela_image.getextrema()
            max_value = max([val[1] for val in band_values])

            # If the maximum value is 0, set it to 1 to avoid division by zero
            if max_value == 0:
                max_value = 1

            # Scale the pixel values of the ELA image to the range [0, 255]
            scale = 255.0 / max_value
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

            # Preprocess the ELA image for the specific model
            img_array = preprocess_image(ela_image, model)

            # Make predictions
            predictions = make_predictions(img_array, model)

            # Delete temporary files
            os.remove('temp.jpg')

            return float(predictions[0])

    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # return jsonify({'message': 'File uploaded successfully'})
        # Instead of returning a JSON response, redirect to the index page
        return render_template('index.html')

    return jsonify({'error': 'Invalid file format'})

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if model_name not in {'xception', 'densenet', 'vgg16'}:
        return jsonify({'error': 'Invalid model name'})

    file = request.files['file']

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the selected model
        if model_name == 'xception':
            model = load_model("model/forgery_detection_model_Xception.h5")
        elif model_name == 'densenet':
            model = load_model("model/forgery_detection_model_DenseNet.h5")  # Replace with your DenseNet model
        elif model_name == 'vgg16':
            model = load_model("model/forgery_detection_model_VGG16.h5")  # Replace with your VGG16 model

        # Make predictions using ELA for the selected model
        model_predictions = apply_ela_and_predict(app.config['UPLOAD_FOLDER'], model)

        # Delete uploaded file
        os.remove(filepath)

        return jsonify({'predictions': model_predictions})

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
