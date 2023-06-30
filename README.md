# Forger Neural Effect on Bela
## Selected Topics in Music and Acoustic Enginering
SELECTED TOPICS IN MUSIC AND ACOUSTIC ENGINEERING
Repository for Selected Topics in Music and Acoustic Enginering 2022/23 Course at Politecnico di Milano

## Abstract
TODO

## Training
TODO 

### Environment Setup 
Conda 
open conda prompt
go to project folder

Deactivate and remove existing environment
conda deactivate
conda env remove -p ./venv

Create new environment
conda env create --prefix ./venv --file environment.yml
conda activate ./venv

### Environment Update 
If you need a new package while developing, update the content of the environment.yml file and run:
```shell script
conda env update --prefix ./venv --file environment.yml  --prune
```

### Config
TODO

### Scripts
TODO
- `train.py`


## Inference

### Bela Cross Compilation 
cross compile:
`cmake -B build -DCMAKE_TOOLCHAIN_FILE:FILEPATH=Toolchain.cmake -DRTNEURAL_XSIMD=ON .`
`cmake --build build -j$(nproc) --target bela_executable`

copy on bela
---------------- Executable ----------------- 
scp build/bin/bela_executable root@192.168.6.2:~/Bela/projects/bela_executable
--------------- Model Weights --------------- 
scp resources/model/model.json root@192.168.6.2:~/Bela/projects/bela_executable
-------------------- GUI -------------------- 
scp src/sketch.js root@192.168.6.2:~/Bela/projects/bela_executable

### Loading on Bela

### Running on Bela