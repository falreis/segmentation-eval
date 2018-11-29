# Lista Comandos

## Ativar Conda
```
source activate i2dl
```

## Gerar Modelo
As opções para geração de modelos estão descritas abaixo.

```
python gen_model-kitti.py --merge=add
python gen_model-kitti.py --merge=avg
python gen_model-kitti.py --merge=max
python gen_model-kitti.py --merge=maj --vote=3
python gen_model-kitti.py --merge=out --out=2
```

### Params
* merge = { avg || max || add || maj || out }
  * Parâmetro correspondente ao nome da operação a ser executada.

* vote? = {1-5} (*OPTIONAL: only for merge==maj*)
  * Parâmetro correspondente ao número de votos necessários para que sejam consideradas as maiorias.

* out? = {1-5} (*OPTIONAL: only for merge==out*)
  * Parâmetro correspondente a saída lateral da rede que será tomada como saída final.

## Run Kitti
As opções para execução das redes neurais estão descritas abaixo.

```
python run-kitti.py --net=hed --merge=add --check=true --load=false
python run-kitti.py --net=hed --merge=avg --check=true --load=false
python run-kitti.py --net=hed --merge=max --check=true --load=false
python run-kitti.py --net=hed --merge=maj --vote=3 --check=true --load=false
python run-kitti.py --net=hed --merge=out --out=2 --check=true --load=false
```

### Params
* net = { hed }
  * Parâmetro correspondente ao nome da rede neural a ser utilizada

* merge = { avg || max || add || maj }
  * Parâmetro correspondente ao nome da operação a ser executada.

* check = {false || true}
  * Parâmetro para armazenamento de pontos de controle.

* vote? = {1-5} (*OPTIONAL: only for merge==maj*)
  * Parâmetro correspondente ao número de votos necessários para que sejam consideradas as maiorias.

* out? = {0-5} (*OPTIONAL: only for merge==maj*)
  * Parâmetro correspondente a saída lateral da rede que será tomada como saída final.

* load? = {false || true} (*INDISPONÍVEL*)
  * *INDISPONÍVEL* - Parâmetro para uso de pesos da rede vggnet16.

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