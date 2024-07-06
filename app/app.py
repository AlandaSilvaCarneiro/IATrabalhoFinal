from flask import Flask, request, render_template, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Diretórios dos modelos
base_dir = 'C:/Users/gutoe/Desktop/IATrabalhoFinal'
upload_dir = os.path.join(base_dir, 'uploads')
primary_model_path = os.path.join(base_dir, 'models/primary_model.h5')
fruit_model_path = os.path.join(base_dir, 'models/fruit_model.h5')
vegetable_model_path = os.path.join(base_dir, 'models/vegetables_model.h5')
package_model_path = os.path.join(base_dir, 'models/packages_model.h5')

# Certifique-se de que o diretório de upload existe
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Carregar modelos
primary_model = tf.keras.models.load_model(primary_model_path)
fruit_model = tf.keras.models.load_model(fruit_model_path)
vegetable_model = tf.keras.models.load_model(vegetable_model_path)
package_model = tf.keras.models.load_model(package_model_path)

# Nomes das classes
primary_classes = ['Fruta', 'Vegetais', 'Pacotes']
fruit_classes = [
    'Abacate', 'Abacaxi', 'Ameixa', 'Banana', 'Kiwi', 'Laranja', 'Lima', 'Limão', 'Maçã', 'Mamão',
    'Manga', 'Maracujá', 'Melão', 'Nectarina', 'Pera', 'Pêssego', 'Romã', 'Satsuma', 'Toranja', 'Melancia'
]
vegetable_classes = [
    'Abobrinha', 'Alho', 'Alho-poró', 'Aspargos', 'Batata', 'Berinjela', 'Beterraba', 'Cebola',
    'Cenouras', 'Cogumelo', 'Gengibre', 'Pepino', 'Pimentão', 'Repolho', 'Tomate'
]
package_classes = [
    'Creme de Leite', 'Iogurte', 'Iogurte de Aveia', 'Iogurte de Soja', 'Leite', 'Leite Azedo',
    'Leite de Aveia', 'Leite de Soja', 'Suco'
]

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalização
    return img_array

# Função para prever a classe
def predict_class(model, img_array, class_names):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence

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
        img_path = os.path.join(upload_dir, file.filename)
        file.save(img_path)

        # Classificação primária
        img_array = load_and_preprocess_image(img_path)
        primary_class, primary_confidence = predict_class(primary_model, img_array, primary_classes)

        # Classificação detalhada
        if primary_class == 'Fruta':
            detailed_class, detailed_confidence = predict_class(fruit_model, img_array, fruit_classes)
        elif primary_class == 'Vegetais':
            detailed_class, detailed_confidence = predict_class(vegetable_model, img_array, vegetable_classes)
        elif primary_class == 'Pacotes':
            detailed_class, detailed_confidence = predict_class(package_model, img_array, package_classes)
        else:
            detailed_class, detailed_confidence = None, None

        result = {
            'category': primary_class,
            'label': detailed_class,
            'probability': detailed_confidence
        }

        return render_template('result.html', result=result)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
