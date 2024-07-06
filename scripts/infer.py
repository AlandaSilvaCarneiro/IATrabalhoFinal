import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Função para carregar e pré-processar a imagem
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalização
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Função para prever a classe
def predict_class(model, img_array, class_names):
    try:
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        return predicted_class, confidence
    except Exception as e:
        print(f"Error predicting class: {e}")
        return None, None

# Função principal de inferência
def infer(model_path, img_path, class_names):
    try:
        model = tf.keras.models.load_model(model_path)
        img_array = load_and_preprocess_image(img_path)
        if img_array is None:
            return None, None
        predicted_class, confidence = predict_class(model, img_array, class_names)
        return predicted_class, confidence
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, None

if __name__ == "__main__":
    img_path = 'C:/Users/gutoe/Desktop/IATrabalhoFinal/path/to/your/image.jpg'  # Atualize este caminho

    # Inferência para a classificação primária
    primary_model_path = 'C:/Users/gutoe/Desktop/IATrabalhoFinal/models/primary_model.h5'  # Atualize este caminho
    primary_class_names = ['Fruta', 'Vegetais', 'Pacotes']
    primary_class, primary_confidence = infer(primary_model_path, img_path, primary_class_names)
    if primary_class is None:
        print("Error in primary classification.")
    else:
        print(f"Primary Classification: {primary_class} with confidence {primary_confidence}")

    # Inferência para classificação detalhada se a classe primária for 'Fruta'
    if primary_class == 'Fruta':
        fruit_model_path = 'C:/Users/gutoe/Desktop/IATrabalhoFinal/models/fruit_model.h5'  # Atualize este caminho
        fruit_class_names = [
            'Abacate', 'Abacaxi', 'Ameixa', 'Banana', 'Kiwi', 'Laranja', 'Lima', 'Limão', 'Maçã', 'Mamão',
            'Manga', 'Maracujá', 'Melão', 'Nectarina', 'Pera', 'Pêssego', 'Romã', 'Satsuma', 'Toranja', 'Melancia'
        ]
        fruit_class, fruit_confidence = infer(fruit_model_path, img_path, fruit_class_names)
        print(f"Fruit Classification: {fruit_class} with confidence {fruit_confidence}")

    # Adicione inferências para Vegetais e Pacotes se necessário
