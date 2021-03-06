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
python kitti.py --func=train --net=alo --merge=avg --check=true --augm=true --load=true --epochs=100 --lr=0.001 --decay=5e-7 --folder=20190102
python kitti.py --func=train --net=alo --merge=max --check=true --augm=true --load=true --epochs=100 --lr=0.001 --decay=5e-7 --folder=20190102
```

### Params
* func = { run }
  * Opção para execução da rede neural.

* net = { slo || alo }
  * Parâmetro correspondente ao nome da rede neural a ser utilizada (slo ou alo).

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

* epochs? = {1-inf} (*OPCIONAL*)
  * Parâmetro para definição do número de épocas para treinamento

* lr? = {0-inf} (*OPCIONAL*)
  * Parâmetro para definição da taxa de aprendizado (*learning rate*)

* decay? = {0-inf} (*OPCIONAL*)
  * Parâmetro para definição do decaimento da taxa de aprendizado (*learning rate decay*)

* folder? = (*OPCIONAL*)
  * Parâmetro para definição da pasta de onde os pesos serão recuperados para predição (subpasta de weights/)


## Test Kitti
As opções para execução das redes neurais estão descritas abaixo.

```
python kitti.py --func=test --net=alo --merge=avg --set=train --mark=false --lr=0.001 --folder=20190102 --morf=true --gray=False
python kitti.py --func=test --net=alo --merge=avg --set=train --mark=false --lr=0.001 --folder=20190102 --morf=true --gray=False
```

### Params
* net = { slo || alo }
  * Parâmetro correspondente ao nome da rede neural a ser utilizada.

* merge = { avg || max || add || maj }
  * Parâmetro correspondente ao nome da operação a ser executada.

* set = {train || test}
  * Parâmetro para conjunto de treino ou teste a ser utilizado.

* mark = {false || true}
  * Parâmetro para marcar rodovia sobre a imagem original ou gerar grond-truth.

* lr? = {0-inf} (*OPCIONAL*)
  * Parâmetro para definição da taxa de aprendizado (*learning rate*)

* folder? = (*OPCIONAL*)
  * Parâmetro para definição da pasta de onde os pesos serão recuperados para predição (subpasta de weights/)

* morf? = {false || true} (*OPCIONAL*)
  * Parâmetro para aplicação de morfologia matemática a predição da rede neural.

* gray? = {false || true} (*OPCIONAL*)
  * Parâmetro para saída de rede em escala cinza.


## Side output Kitti
As opções para execução de saídas laterais (somente para visualização) estão descritas abaixo.

python kitti.py --func=side --net=alo --merge=avg --set=train --mark=false --lr=0.001 --folder=20190102

### Params
* net = { slo || alo }
  * Parâmetro correspondente ao nome da rede neural a ser utilizada.

* merge = { avg || max || add || maj }
  * Parâmetro correspondente ao nome da operação a ser executada.

* set = {train || test}
  * Parâmetro para conjunto de treino ou teste a ser utilizado.

* mark = {false || true}
  * Parâmetro para marcar rodovia sobre a imagem original ou gerar grond-truth.

* lr? = {0-inf} (*OPCIONAL*)
  * Parâmetro para definição da taxa de aprendizado (*learning rate*)

* folder? = (*OPCIONAL*)
  * Parâmetro para definição da pasta de onde os pesos serão recuperados para predição (subpasta de weights/)

---
## Eval Kitti

### Change File Names

To eval the results, change the name of the files create by the test procedure. Following this tutorial, it is necessary to install *rename* program, as the code below.

```
sudo apt install rename
cd ../export/Kitti/slo/train
rename 's/umm_/umm_road_/' * 
rename 's/um_/um_road_/' * 
rename 's/uu_/uu_road_/' *
```

### Eval Procedure

The commands should have the same structure as the code below.

```
cd ../../eval/Kitti/
python evaluateRoad.py ../../code/export/Kitti/slo/train ../../code/export
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