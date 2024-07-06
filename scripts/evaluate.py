import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Função para avaliar o modelo
def evaluate_model(model_path, validation_dir, class_names, image_size=(224, 224), batch_size=32):
    model = tf.keras.models.load_model(model_path)

    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_names
    )

    loss, accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
    print(f'Model: {model_path} - Loss: {loss}, Accuracy: {accuracy}')

# Exemplo de uso
if __name__ == "__main__":
    base_validation_dir = 'C:/Users/gutoe/Desktop/IATrabalhoFinal/datasets/validation'

    evaluate_model('C:/Users/gutoe/Desktop/IATrabalhoFinal/models/primary_model.h5', base_validation_dir, [
        'Fruta', 'Vegetais', 'Pacotes'
    ])

    evaluate_model('C:/Users/gutoe/Desktop/IATrabalhoFinal/models/fruit_model.h5', base_validation_dir + '/Fruta', [
        'Abacate', 'Abacaxi', 'Ameixa', 'Banana', 'Kiwi', 'Laranja', 'Lima', 'Limão', 'Maçã', 'Mamão',
        'Manga', 'Maracujá', 'Melão', 'Nectarina', 'Pera', 'Pêssego', 'Romã', 'Satsuma', 'Toranja', 'Melancia'
    ])

    evaluate_model('C:/Users/gutoe/Desktop/IATrabalhoFinal/models/vegetables_model.h5', base_validation_dir + '/Vegetais', [
        'Abobrinha', 'Alho', 'Alho-poró', 'Aspargos', 'Batata', 'Berinjela', 'Beterraba', 'Cebola',
        'Cenouras', 'Cogumelo', 'Gengibre', 'Pepino', 'Pimentão', 'Repolho', 'Tomate'
    ])

    evaluate_model('C:/Users/gutoe/Desktop/IATrabalhoFinal/models/packages_model.h5', base_validation_dir + '/Pacotes', [
        'Creme de Leite', 'Iogurte', 'Iogurte de Aveia', 'Iogurte de Soja', 'Leite', 'Leite Azedo',
        'Leite de Aveia', 'Leite de Soja', 'Suco'
    ])
