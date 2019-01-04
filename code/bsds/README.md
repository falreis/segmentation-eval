# Lista Comandos

## Ativar Conda
```
source activate i2dl
```

## Npy BSDS
As opções para condensação de imagens em um único arquivo npy (melhora de desempenho ao fazer io a cada execução)

```
python bsds.py --func=npy --set=train --augm=true
```

### Params
* set = {train || test}
  * Parâmetro para conjunto de treino ou teste a ser utilizado.

* augm? = {false || true}
  * Parâmetro para uso de dataset com data augmentation.


## Train BSDS
As opções para execução das redes neurais estão descritas abaixo.

```
python bsds.py --func=train --net=alo --merge=max --check=true --augm=true --load=true --epochs=100 --lr=0.001 --folder=20181211
python bsds.py --func=train --net=slo --merge=max --check=true --augm=true --load=true --epochs=100 --lr=0.001 --folder=20181211
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

* folder? = (*OPCIONAL*)
  * Parâmetro para definição da pasta de onde os pesos serão recuperados para predição (subpasta de weights/)


## Test BSDS
As opções para execução das redes neurais estão descritas abaixo.

```
python bsds.py --func=test --net=slo --merge=max --set=train --mark=false --lr=0.001 --folder=20181211 --morf=true --gray=False
python bsds.py --func=test --net=alo --merge=max --set=train --mark=false --lr=0.001 --folder=20181211 --morf=true --gray=False
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

---
## Eval BSDS

### Change File Names

To eval the results, change the name of the files create by the test procedure. Following this tutorial, it is necessary to install *rename* program, as the code below.

```
sudo apt install rename
cd ../export/BSDS/slo/train
rename 's/umm_/umm_road_/' * 
rename 's/um_/um_road_/' * 
rename 's/uu_/uu_road_/' *
```

### Eval Procedure

The commands should have the same structure as the code below.

```
cd ../../eval/BSDS/
python evaluateRoad.py ../../code/export/BSDS/slo/train ../../code/export
```

---

## Help (SCP)
Home > Server
```
scp gen_small-bsds.py falreis@laplace.icei.pucminas.br:/home/falreis/machine_learning/code/AMRP/gen_small-bsds.py
```

Server > Home
```
scp falreis@laplace.icei.pucminas.br:/home/falreis/machine_learning/code/AMRP/gen_small-bsds.py gen_small-bsds.py
```