# Adaptação da Rede HED para Segmentação de Imagens

## Download das bases de dados

Faça download das bases de dados seguindos os tutoriais para cada uma delas:
* [CamVid](data/CamVid/) 
* [KITTI](data/Kitti/) 

## Resultados das bases de dados

## Pré processamento das bases

## Treinamento

### Teste 1: 30/11/2018 - Base de Dados com Data Augmentation

#### Operações:
- maj2
- maj3
- max
- add
- avg

#### Pesos Iniciais

vgg16 - 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

#### Parâmetros
```
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.95, nesterov=False)
net_basic.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
net_basic.compile(loss=Vote, optimizer=sgd, metrics=["accuracy"])
```

- otimizador: SGD
- learning rate: 0.0001
- decay: 1e-6
- momentum: 0.95
- nesterov: Não
- perda: 'categorical_crossentropy', para max, add e avg, e 'vote', para maj2 e maj3.
- métrica: acurácia
- validation_split=0.15 (15%)
- shuffle: sim
- amostras: 1228 de treino, 217 de validação
- épocas: 400
- batch size: 10
- armazenamento de pesos: somente melhor acurácia


### Teste 2: 01/12/2018 - Base de Dados sem Data Augmentation

#### Operações:
- maj2
- maj3
- max
- add
- avg

#### Pesos Iniciais

vgg16 - 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

#### Parâmetros
```
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.95, nesterov=False)
net_basic.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
```

- otimizador: SGD
- learning rate: 0.0001
- decay: 1e-6
- momentum: 0.95
- nesterov: Não
- perda: 'categorical_crossentropy', para max, add e avg, e 'vote', para maj2 e maj3.
- métrica: acurácia
- validation_split=0.15 (15%)
- shuffle: sim
- amostras: 245 de treino, 44 de validação
- épocas: 400
- batch size: 10
- armazenamento de pesos: somente melhor acurácia

## Avaliação de Desempenho