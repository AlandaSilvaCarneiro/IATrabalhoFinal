from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Definir caminhos dos modelos (atualize conforme necessário)
MODEL_PATHS = {
    'primary': 'path/to/models/model_primary.h5',
    'fruit': 'path/to/models/model_fruit.h5',
    'vegetables': 'path/to/models/model_vegetables.h5',
    'packages': 'path/to/models/model_packages.h5'
}

# Carregar modelos
models = {
    'primary': load_model(MODEL_PATHS['primary']),
    'fruit': load_model(MODEL_PATHS['fruit']),
    'vegetables': load_model(MODEL_PATHS['vegetables']),
    'packages': load_model(MODEL_PATHS['packages'])
}

def prepare_image(image, target_size):
    try:
        image = image.resize(target_size)
        image = image.convert('RGB')  # Converter para RGB
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_label_from_prediction(prediction, labels):
    return labels[np.argmax(prediction)]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('static/uploads', filename)

        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')

        file.save(file_path)
        image = Image.open(file_path)

        prepared_image = prepare_image(image, target_size=(224, 224))
        if prepared_image is None:
            return redirect(request.url)

        primary_prediction = models['primary'].predict(prepared_image)
        primary_label = get_label_from_prediction(primary_prediction, ['Fruta', 'Vegetal', 'Pacote'])

        if primary_label == 'Fruta':
            specific_prediction = models['fruit'].predict(prepared_image)
            specific_label = get_label_from_prediction(specific_prediction, os.listdir('path/to/data/train/images/Fruta'))
        elif primary_label == 'Vegetal':
            specific_prediction = models['vegetables'].predict(prepared_image)
            specific_label = get_label_from_prediction(specific_prediction, os.listdir('path/to/data/train/images/Vegetais'))
        else:
            specific_prediction = models['packages'].predict(prepared_image)
            specific_label = get_label_from_prediction(specific_prediction, os.listdir('path/to/data/train/images/Pacotes'))

        result = {
            "category": primary_label,
            "label": specific_label,
            "probability": float(np.max(specific_prediction))
        }
        return render_template('result.html', result=result, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
