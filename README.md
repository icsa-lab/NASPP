#  Release of **Neural Architecture Search for Portrait Parsing**

This is a notebook for **Neural Architecture Search for Portrait Parsing**
In this notebook, we will demonstrate:

- How to reconstruct the parsing network by the decoder configuration that is represented by the genotypes. (Macro-structure configuration and the micro-structure configuration).
- The inference results of parsing models on the datasets (EG1800, HELEN, CelebAMask-HQ).

For reproduction of our searched model, the Ubuntu OS is recommended. The models have been tested using Python 3.6 +  Pytorch 1.0.0

### Required packages:

```
cv2
jupyter-notebook
matplotlib
numpy
Pillow
torch>=1.0
torchvision
thop
```

## Getting Started
```bash
pip install --upgrade pip
pip install --upgrade jupyter notebook
```

Then, please clone this repository to your computer using:

```bash
git clone https://github.com/icsa-lab/NASPP
```

After the cloning, you may go to the directory and run:

```bash
jupyter notebook --port 8888
```

to start a jupyter notebook and explore the  `inference.ipynb` prepared by us!