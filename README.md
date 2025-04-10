---
Authors: Gustav Emil Mark-Hansen, Frederik Henriques Altmann, Christian Igel & Ankit Kariryaa
---
# Demosaicing and Neural Network Based Image Reconstruction in the Presence of Noise

Code for generating CFA data, and training U-Net models.


### Models
Denoising U-net model: [notebooks/denoisingunet.py](notebooks/denoisingunet.py)

Training loop: [notebooks/train.py](notebooks/train.py)

### Datasets
noise augmentation + .dng convertion: [notebooks/ds_augment.py](notebooks/ds_augment.py)

.dng convertion: [notebooks/ds_convert.py](notebooks/ds_convert.py)

### Misc.
Slurm specific scripts: [workarounds/](workarounds/)
