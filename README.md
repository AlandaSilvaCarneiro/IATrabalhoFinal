# Projeto de Reconhecimento de Objetos em Imagens para Segmentação de Compras

## Descrição

Este projeto utiliza redes neurais convolucionais (CNNs) para classificar imagens de produtos em três categorias principais: frutas, vegetais e pacotes. Após a classificação primária, uma classificação detalhada é feita para identificar o tipo específico de fruta, vegetal ou pacote.

## Estrutura do Projeto

IATrabalhoFinal/
├── app/
│ ├── static/
│ │ └── styles.css
│ ├── templates/
│ │ ├── index.html
│ │ └── result.html
│ ├── app.py
│ └── requirements.txt
├── datasets/
│ ├── train/
│ │ ├── fruit/
│ │ ├── packages/
│ │ ├── primary/
│ │ └── vegetables/
│ ├── validation/
│ │ ├── fruit/
│ │ ├── packages/
│ │ ├── primary/
│ │ └── vegetables/
│ └── test/
│ ├── fruit/
│ ├── packages/
│ ├── primary/
│ └── vegetables/
├── models/
│ └── (modelos treinados serão salvos aqui)
├── notebooks/
│ └── Treinamento_de_Modelos.ipynb
├── scripts/
│ ├── train_fruit.py
│ ├── train_packages.py
│ ├── train_primary.py
│ ├── train_vegetables.py
│ └── evaluate_models.py
└── README.md
## Configuração do Ambiente

### Requisitos

- Python 3.10 ou superior

### Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/AlandaSilvaCarneiro/IATrabalhoFinal.git
   cd IATrabalhoFinal

2. Crie um ambiente virtual (opcional, mas recomendado):
python -m venv venv
venv\Scripts\activate  # No Windows
source venv/bin/activate  # No Linux/Mac

3.Instale as dependências:
pip install -r requirements.txt
 
### Treinamento dos Modelos

 Os scripts para treinamento dos modelos estão localizados na pasta scripts. Você pode executar cada script individualmente ou utilizar o notebook Treinamento_de_Modelos.ipynb para treinar todos os modelos.

### Treinar um Modelo Individualmente

python scripts/train_fruit.py
python scripts/train_packages.py
python scripts/train_primary.py
python scripts/train_vegetables.py

### Treinar Usando o Notebook

Abra o notebook Treinamento_de_Modelos.ipynb e execute as células para treinar todos os modelos.
 
### Avaliação dos modelos

Após o treinamento, você pode avaliar os modelos utilizando o notebook Avaliacao_de_Modelos.ipynb.

### Executar a Aplicação Flask

A aplicação Flask permite fazer upload de imagens e obter a classificação.

1. Navegue até a pasta app:
cd app

2. Execute a aplicação Flask:
python app.py

3.Acesse http://127.0.0.1:5000 no seu navegador e faça upload de uma imagem para obter a classificação.

### Contribuição

Contribuições são bem-vindas! Por favor, abra um issue ou envie um pull request para discutir mudanças.

### Licença

Este projeto está licenciado sob a MIT License.



