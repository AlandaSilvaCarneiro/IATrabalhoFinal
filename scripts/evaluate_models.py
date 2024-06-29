import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, test_dir):
    model = tf.keras.models.load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

    loss, accuracy = model.evaluate(test_generator)
    print(f'Test accuracy: {accuracy}, Test loss: {loss}')

if __name__ == "__main__":
    # Avaliando o modelo de frutas
    model_path = 'models/fruit_model.h5'
    test_dir = 'datasets/test/fruit'
    print("Evaluating fruit model...")
    evaluate_model(model_path, test_dir)
    
    # Avaliando o modelo de vegetais
    model_path = 'models/vegetables_model.h5'
    test_dir = 'datasets/test/vegetables'
    print("Evaluating vegetables model...")
    evaluate_model(model_path, test_dir)
    
    # Avaliando o modelo de pacotes
    model_path = 'models/packages_model.h5'
    test_dir = 'datasets/test/packages'
    print("Evaluating packages model...")
    evaluate_model(model_path, test_dir)
    
    # Avaliando o modelo primário
    model_path = 'models/primary_model.h5'
    test_dir = 'datasets/test/primary'
    print("Evaluating primary model...")
    evaluate_model(model_path, test_dir)
