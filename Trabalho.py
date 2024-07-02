import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Função para carregar e preprocessar as imagens
def carregar_imagens(caminhos_imagens, tamanho_img=(100, 100)):
    imagens = []
    for caminho in caminhos_imagens:
        img = tf.keras.preprocessing.image.load_img(caminho, target_size=tamanho_img)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        imagens.append(img_array)
    imagens = np.array(imagens)
    return imagens

# Caminho do modelo salvo
caminho_modelo = 'modelo_completo.h5'

# Carregar o modelo
modelo = load_model(caminho_modelo)

# Caminhos das novas imagens para predição
caminhos_imagens = [
'./Test/testeClasses/apple/1_100.jpg',
    './Test/testeClasses/apple/12_100.jpg',

]

# Carregar e preprocessar as novas imagens
tamanho_img = (100, 100)
imagens = carregar_imagens(caminhos_imagens, tamanho_img)

# Fazer predições nas novas imagens
predicoes = modelo.predict(imagens)
classes_preditas = np.argmax(predicoes, axis=1)

# Mapeamento de índices para nomes de classes
nome_classes = ["Apple", "Limes", "Pinaapple", "Tomate"]

# Mostrar as predições com nomes de classes
for i, caminho in enumerate(caminhos_imagens):
    print(f"Imagem: {caminho} - Classe Predita: {nome_classes[classes_preditas[i]]}")

# Para este exemplo, assumimos que as classes verdadeiras são conhecidas
classes_verdadeiras = [0,0]  # Substitua pelas classes verdadeiras correspondentes

# Calcular e imprimir métricas
accuracy = accuracy_score(classes_verdadeiras, classes_preditas)
f1 = f1_score(classes_verdadeiras, classes_preditas, average='weighted')
print(f"Acurácia: {accuracy}")
print(f"F1-Score: {f1}")

# Matriz de Confusão
cm = confusion_matrix(classes_verdadeiras, classes_preditas)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nome_classes, yticklabels=nome_classes)
plt.xlabel('Classe Predita')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()

# Relatório de Classificação
relatorio = classification_report(classes_verdadeiras, classes_preditas, target_names=nome_classes)
print('Relatório de Classificação:\n', relatorio)

# Histórico do Treinamento
historico = modelo.history.history

# Gráfico de Acurácia
plt.plot(historico['accuracy'], label='Acurácia de Treinamento')
plt.plot(historico['val_accuracy'], label='Acurácia de Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(loc='lower right')
plt.title('Acurácia de Treinamento e Validação')
plt.show()

# Gráfico de Perda
plt.plot(historico['loss'], label='Perda de Treinamento')
plt.plot(historico['val_loss'], label='Perda de Validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend(loc='upper right')
plt.title('Perda de Treinamento e Validação')
plt.show()
