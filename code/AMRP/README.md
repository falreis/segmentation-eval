# Lista Comandos

## Ativar Conda
```
source activate i2dl
```

## Run Kitti
As opções para execução das redes neurais estão descritas abaixo.

```
python kitti.py --func=run --net=full --check=true --augm=true --load=true
python kitti.py --func=run --net=hed --merge=add --check=true --augm=true --load=true
python kitti.py --func=run --net=hed --merge=avg --check=true --augm=true --load=true
python kitti.py --func=run --net=hed --merge=max --check=true --augm=true --load=true
python kitti.py --func=run --net=hed --merge=maj --check=true --augm=true --load=true --vote=3
```

### Params
* func = { run }
  * Opção para execução da rede neural.

* net = { hed || full }
  * Parâmetro correspondente ao nome da rede neural a ser utilizada (hed ou full).

* merge = { avg || max || add || maj }
  * Parâmetro correspondente ao nome da operação a ser executada.

* check = {false || true}
  * Parâmetro para armazenamento de pontos de controle.

* vote? = {1-5} (*OPCIONAL: somente para merge==maj*)
  * Parâmetro correspondente ao número de votos necessários para que sejam consideradas as maiorias.

* augm? = {false || true}  (*OPCIONAL*)
  * Parâmetro para uso de dataset com data augmentation.

* load? = {false || true} (*OPCIONAL*)
  * Parâmetro para uso de pesos da rede vggnet16.


## Eval Kitti
As opções para execução das redes neurais estão descritas abaixo.

```
python kitti.py --func=eval --net=hed --merge=max --set=train --mark=false
python kitti.py --func=eval --net=full --set=train --mark=false
```

### Params
* net = { hed || full }
  * Parâmetro correspondente ao nome da rede neural a ser utilizada.

* merge = { avg || max || add || maj }
  * Parâmetro correspondente ao nome da operação a ser executada.

* set = {train || test}
  * Parâmetro para conjunto de treino ou teste a ser utilizado.

* mark = {false || true}
  * Parâmetro para marcar rodovia sobre a imagem original ou gerar grond-truth.


## Npy Kitti
As opções para condensação de imagens em um único arquivo npy (melhora de desempenho ao fazer io a cada execução)

```
python kitti.py --func=npy --set=train --augm=true
```

### Params
* set = {train || test}
  * Parâmetro para conjunto de treino ou teste a ser utilizado.

* augm? = {false || true}
  * Parâmetro para uso de dataset com data augmentation.

---
---
## Help (SCP)
Home > Server
```
scp gen_small-kitti.py falreis@laplace.icei.pucminas.br:/home/falreis/machine_learning/code/AMRP/gen_small-kitti.py
```

Server > Home
```
scp falreis@laplace.icei.pucminas.br:/home/falreis/machine_learning/code/AMRP/gen_small-kitti.py gen_small-kitti.py
```