# StyCona

This repo is the PyTorch implementation for the paper:

**["Style Content Decomposition-based Data Augmentation for Domain Generalizable Medical Image Segmentation"](https://proceedings.mlr.press/v315/shen26a.html)** 

## Usage

### 0. Requirements
The code is developed using Python 3.8 with PyTorch 1.11.0.
All experiments in our paper were conducted on a single NVIDIA A40 GPU with 48G GPU memory.

Install the main packages:
```angular2html
pytorch == 1.11.0
torchvision == 0.12.0
```

### 1. Data Preparation
#### 1.1. Download data
The datasets can be downloaded from the following links:
* MSCMR Dataset - [Link](https://zmiclab.github.io/zxh/0/mscmrseg19/index.html) 
* Fundus Dataset - [Link](https://zenodo.org/records/8009107)

PS: *Please cite the original dataset papers when using these datasets in your publications.*


#### 1.2. Split Dataset
Follow the list files (within the "*data*" folders) to split the datasets

### 2. Training
```angular2html
python train_stycona.py
```

### 3. Evaluation
```angular2html
python eval.py
```


## Citation
If you find this project useful, please consider citing:
```
@inproceedings{shen2026style,
  title = {Style Content Decomposition-based Data Augmentation for Domain Generalizable Medical Image Segmentation},
  author = {Shen, Zhiqiang and Cao, Peng and Yang, Jinzhu and Zaiane, Osmar R. and Chen, Zhaolin},
  booktitle = {Proceedings of The 9th International Conference on Medical Imaging with Deep Learning},
  volume = {315},
  pages = {21--35},
  year = {2026},
  publisher = {PMLR}
}
```


## Contact
If you have any questions or suggestions, please feel free to contact me ([xxszqyy@gmail.com](xxszqyy@gmail.com)).


## Acknowledgements
* [TriD](https://github.com/Chen-Ziyang/TriD)

Thanks to the authors for providing the processed data.
