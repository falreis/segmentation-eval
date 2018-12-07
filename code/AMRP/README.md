# Lista Comandos

## Ativar Conda
```
source activate i2dl
```

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


## Train Kitti
As opções para execução das redes neurais estão descritas abaixo.

```
python kitti.py --func=train --net=full --check=true --augm=true --load=true --epochs=100 --balanced=False
python kitti.py --func=train --net=hed --merge=add --check=true --augm=true --load=true --epochs=100 --balanced=False
python kitti.py --func=train --net=hed --merge=avg --check=true --augm=true --load=true --epochs=100 --balanced=False
python kitti.py --func=train --net=hed --merge=max --check=true --augm=true --load=true --epochs=100 --balanced=False
python kitti.py --func=train --net=hed --merge=maj --check=true --augm=true --load=true --epochs=100 --vote=3 --balanced=False
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

* epochs? = {1-*} (*OPCIONAL*)
  * Parâmetro para definição do número de épocas para treinamento

* balanced? = {false || true}} (*OPCIONAL*)
  * Parâmetro para definição de utilização de função *categorical_crossentropy* ou *balanced_categorical_crossentropy* (própria)


## Test Kitti
As opções para execução das redes neurais estão descritas abaixo.

```
python kitti.py --func=test --net=hed --merge=max --set=train --mark=false --morf=true
python kitti.py --func=test --net=full --set=train --mark=false --morf=true
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

* morf? = {false || true} (*OPCIONAL*)
  * Parâmetro para aplicação de morfologia matemática a predição da rede neural.

---
## Eval Kitti

### Change File Names

To eval the results, change the name of the files create by the test procedure. Following this tutorial, it is necessary to install *rename* program, as the code below.

```
sudo apt install rename
cd ../export/Kitti/hed/train
rename 's/umm_/umm_road_/' * 
rename 's/um_/um_road_/' * 
rename 's/uu_/uu_road_/' *
```

### Eval Procedure

The commands should have the same structure as the code below.

```
cd ../../eval/Kitti/
python evaluateRoad.py ../../code/export/Kitti/hed/train ../../code/export
```

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