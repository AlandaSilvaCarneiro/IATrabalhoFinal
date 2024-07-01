import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score
import numpy as np

# Função para criar o modelo AlexNet adaptado para CIFAR-10
def AlexNet(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Função para filtrar as classes desejadas
def filter_classes(x, y, classes):
    indices = np.isin(y, classes).flatten()
    x_filtered = x[indices]
    y_filtered = y[indices]
    return x_filtered, y_filtered

# Carregar o conjunto de dados CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Classes desejadas: 0 (avião), 1 (automóvel), 9 (caminhão)
desired_classes = [0, 1, 9]

# Filtrar as classes desejadas no conjunto de treinamento e teste
x_train_filtered, y_train_filtered = filter_classes(x_train, y_train, desired_classes)
x_test_filtered, y_test_filtered = filter_classes(x_test, y_test, desired_classes)

# Mapeamento dos rótulos para novas classes 0, 1, 2
class_mapping = {0: 0, 1: 1, 9: 2}
y_train_filtered = np.vectorize(class_mapping.get)(y_train_filtered)
y_test_filtered = np.vectorize(class_mapping.get)(y_test_filtered)

# Convertendo rótulos para formato one-hot
y_train_filtered = to_categorical(y_train_filtered, num_classes=len(desired_classes))
y_test_filtered = to_categorical(y_test_filtered, num_classes=len(desired_classes))

# Parâmetros
input_shape = (32, 32, 3)
num_classes = len(desired_classes)
epochs = 3
batch_size = 32

# Criar o modelo
model = AlexNet(input_shape=input_shape, num_classes=num_classes)

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train_filtered, y_train_filtered, validation_data=(x_test_filtered, y_test_filtered), epochs=epochs, batch_size=batch_size)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test_filtered, y_test_filtered)
print(f"Test accuracy: {test_acc}")

# Calcular o F1-Score
y_true = np.argmax(y_test_filtered, axis=1)
y_pred = np.argmax(model.predict(x_test_filtered), axis=1)

f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1-Score: {f1:.4f}")

# Fazer predições em novas imagens (exemplo)
predictions = model.predict(x_test_filtered[:5])
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_filtered[:5], axis=1)

print(f"Predicted classes: {predicted_classes}")
print(f"True classes: {true_classes}")

