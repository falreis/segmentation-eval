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
* merge = { avg || max || add || maj }
* vote? = {1-5} (*OPTIONAL: only for merge==maj*)
* out? = {0-5} (*OPTIONAL: only for merge==maj*)

## Run Kitti
```
python run-kitti.py --net=hed --merge=maj --vote=2 --check=false --out=2
```

### Params
* net = { hed || rcf }
* merge = { avg || max || add || maj }
* check = {false || true} (*OPTIONAL: recover best checkpoint*)
* vote? = {1-5} (*OPTIONAL: only for merge==maj*)
* out? = {0-5} (*OPTIONAL: only for merge==maj*)