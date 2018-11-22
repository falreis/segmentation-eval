# Adaptação da Rede HED para Segmentação de Imagens

## Código Fonte

Código fonte do projeto.
Para configuração do ambiente virtual, importe o arquivo Anaconda [i2dl.yml](code/i2dl.yml), conforme instruções abaixo. O ambiente para avaliação somente é executado com Python 3.

conda env create -f i2dl.yml


O passo a passo para execução da rede neural já treinada, treinamento para as bases de dados CamVid e Kitti e avaliação de desempenho está disponível no tutorial do [código fonte](code/).

## Avaliação de Desempenho das Redes

Código fonte para avaliação da base de dados KITTI.
Para configuração do ambiente virtual, importe o arquivo Anaconda [kitti.yml](eval/kitti.yml), conforme instruções abaixo. O ambiente para avaliação somente é executado com Python 2.

```
conda env create -f kitty.yml
```

## Papers

Disponível na pasta papers.

O artigo para a disciplina de Aprendizado de Máquina e Reconhecimento de Padrões está no diretório papers/amrp.

---
## Useful information
Mini tutorial para acesso a documentos na Amazon.

```
ssh -i "deep_learning.pem" ec2-user@ec2-54-219-182-179.us-west-1.compute.amazonaws.com
```

```
scp -i "deep_learning.pem" segmentation-eval/code/data/Kitti/test.tar.gz ec2-user@ec2-54-153-49-79.us-west-1.compute.amazonaws.com:~/segmentation-eval/code/data/Kitti/

```

