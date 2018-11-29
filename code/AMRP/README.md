# Lista Comandos

## Ativar Conda
```
source activate i2dl
```

## Gerar Modelo
```
python gen_model-kitti.py --merge=maj --vote=2 --out=2
```

### Params
* merge = { avg || max || add || maj || out }
* vote? = {1-5} (*OPTIONAL: only for merge==maj*)
* out? = {1-5} (*OPTIONAL: only for merge==out*)

## Run Kitti
```
python run-kitti.py --net=hed --merge=out --vote=2 --check=false --out=2
```

### Params
* net = { hed || rcf }
* merge = { avg || max || add || maj }
* check = {false || true} (*OPTIONAL: recover best checkpoint*)
* vote? = {1-5} (*OPTIONAL: only for merge==maj*)
* out? = {0-5} (*OPTIONAL: only for merge==maj*)

## Other code (SCP)
```
scp gen_small-kitti.py falreis@laplace.icei.pucminas.br:/home/falreis/machine_learning/code/AMRP/gen_small-kitti.py
```
