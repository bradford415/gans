# Gans
Modular GAN framework, mostly my own learning.

## Setup
For windows run:
```bash
make create_windows
```
For linux run:
```bash
make create
```

## DCGAN
Deep convolutional GAN implementation

### Train
```bash
python scripts/train.py scripts/configs/base-config.yaml scripts/configs/dcgan/dcgan-base.yaml
```

### Results
Results after `30 epochs`

![epoch029](https://github.com/bradford415/gans/assets/34605638/3107812f-12c7-4015-9270-9d778d2c9529)

### Resources
- DCGAN Paper: https://arxiv.org/abs/1511.06434
- DCGAN: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
