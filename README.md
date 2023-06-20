# CLAR: Contrastive Learning of Auditory Representations

###  This repository contains an unofficial pytorch implementation of the paper CLAR:Contrastive Learning of Auditory Representations. [Paper Link](https://arxiv.org/abs/2010.09542)


# Install requirements
Before running the scripts, install conda, requirements and activate the environment env
```
$ conda create --name env --file requirements.txt
$ conda activate env
```

# Dataset
The paper CLAR adopted three different datasets: Speech Commands, NSynth, and Environmental Sound Classification. However, this repository only utilizes the Speech Commands dataset, which can be easily downloaded through the PyTorch Audio library. Therefore, the repository focuses on implementing and applying the CLAR method specifically to the **Speech Commands** dataset.

# CLAR Architecture
![method](img/methods.png)

# Augmentation
The paper CLAR studied various augmentations, including Pitch Shift (High and Low steps), Fade In/Out, Time Masking, Time Shift, Time Stretch (Low and High Rate), and Noise (White, Pink, Brown). In this implementation, three augmentations are presented: *Fade In/Out, White Noise, and Time Masking*. To add other augmentations, you should insert them into the **augmentation.py** file.



