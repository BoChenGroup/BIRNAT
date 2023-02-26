
# Bidirectional Recurrent Neural Networks with Adversarial Training (BIRNAT)

This repository contains the code for the paper **BIRNAT: Bidirectional Recurrent Neural Networks with Adversarial Training for Video Snapshot Compressive Imaging** (***The European Conference on Computer Vision*** 2020) by [Ziheng Cheng](https://github.com/zihengcheng), Ruiying Lu, Zhengjue Wang, Hao Zhang, [Bo Chen](https://web.xidian.edu.cn/bchen/), Ziyi Meng and [Xin Yuan](https://www.bell-labs.com/usr/x.yuan).

## Requirements
```
PyTorch > 1.3.0
numpy
scipy
```

### Docker
```
docker pull bryanbocao/birnat
docker run -d --ipc=host --shm-size=16384m -it -v /:/share --gpus all --network=bridge bryanbocao/birnat /bin/bash
```
Check <CONTAINER_ID>
```
docker ps -a
```
e.g.
```
CONTAINER ID   IMAGE               COMMAND       CREATED         STATUS                       PORTS     NAMES
<CONTAINER_ID> bryanbocao/birnat   "/bin/bash"   6 seconds ago   Up 3 seconds                           xenodochial_herschel
```
Enter 
```
docker exec -it <CONTAINER_ID> /bin/bash
```
```
cd <PATH_TO_BIRNAT>
```

## Data
The training data for BIRNAT is generated from [DAVIS2017](https://davischallenge.org/davis2017/code.html) with random crop and data argumentation and final obtain 26000 data pairs. If you want to use the same training data as ours, please run ```training_data / data_generation.m``` in MATLAB (for simulated data, we use 480p resolution; for real data, we use 1080p resolution).

The simulation test data includes six simulation data in the ```simulation_test``` folder. Three real data reconstructions for BIRNAT are in the ```result/real``` folder.

## Train
Run model without adversarial training:
```
python train.py
```
Run model with adversarial training:
```
python train_at.py
```

The adversarial training and discriminator reference [this](https://github.com/LMescheder/GAN_stability). Note that running model without adversarial training requires more than 27GB of memory and with adversarial training need 32GB which batch size is 3. Please make sure your GPU is available.

## Test
Run
```
python test.py
```

where will evaluate the preformance on simulation data using the pre-trained model in ```model/```.


## Citation
```
@inproceedings{Cheng20ECCV_BIRNAT,
author = {Cheng, Ziheng and Lu, Ruiying and Wang, Zhengjue and Zhang, Hao and Chen, Bo and Meng, Ziyi and Yuan, Xin},
title = {{BIRNAT}: Bidirectional Recurrent Neural Networks with Adversarial Training for Video Snapshot Compressive Imaging},
booktitle = {European Conference on Computer Vision (ECCV)},
month = {August},
year = {2020}
}
```


## Contact
[Ziheng Cheng, Xidian University](mailto:zhcheng@stu.xidian.edu.cn "Ziheng Cheng, Xidian University") 

[Bo Chen, Xidian University](mailto:bchen@mail.xidian.edu.cn "Bo Chen, Xidian University") 

[Xin Yuan, Bell Labs](mailto:xyuan@bell-labs.com "Xin Yuan, Bell labs")  



































