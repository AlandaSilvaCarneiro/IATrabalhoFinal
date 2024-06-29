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
    img_path = 'path/to/your/image.jpg'  # Atualize este caminho

    # Inferência para a classificação primária
    primary_model_path = 'path/to/models/model_primary.h5'  # Atualize este caminho
    primary_class_names = ['Fruta', 'Vegetais', 'Pacotes']
    primary_class, primary_confidence = infer(primary_model_path, img_path, primary_class_names)
    if primary_class is None:
        print("Error in primary classification.")
        exit(1)

    print(f'Primary Classification: {primary_class} ({primary_confidence:.4f})')

    if primary_class == 'Fruta':
        model_path = 'path/to/models/model_fruit.h5'  # Atualize este caminho
        class_names = ['Abacate', 'Abacaxi', 'Ameixa', 'Banana', 'Kiwi', 'Laranja', 'Lima', 'Limão', 'Maçã', 'Mamão',
                       'Manga', 'Maracujá', 'Melão', 'Nectarina', 'Pera', 'Pêssego', 'Romã', 'Satsuma', 'Toranja']
    elif primary_class == 'Vegetais':
        model_path = 'path/to/models/model_vegetables.h5'  # Atualize este caminho
        class_names = ['Abobrinha', 'Alho', 'Alho-poró', 'Aspargos', 'Batata', 'Berinjela', 'Beterraba', 'Cebola',
                       'Cenouras', 'Cogumelo', 'Gengibre', 'Pepino', 'Pimentão', 'Repolho', 'Tomate']
    elif primary_class == 'Pacotes':
        model_path = 'path/to/models/model_packages.h5'  # Atualize este caminho
        class_names = ['Creme de Leite', 'Iogurte', 'Iogurte de Aveia', 'Iogurte de Soja', 'Leite', 'Leite Azedo',
                       'Leite de Aveia', 'Leite de Soja', 'Suco']
    else:
        print("Unknown primary classification.")
        exit(1)

    specific_class, specific_confidence = infer(model_path, img_path, class_names)
    if specific_class is None:
        print("Error in specific classification.")
        exit(1)

    print(f'Specific Classification: {specific_class} ({specific_confidence:.4f})')
