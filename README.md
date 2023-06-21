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
The paper CLAR studied various augmentations, including Pitch Shift (High and Low steps), Fade In/Out, Time Masking, Time Shift, Time Stretch (Low and High Rate), and Noise (White, Pink, Brown). In this implementation, three augmentations are presented: *Fade In/Out, White Noise, and Time Masking*. To add other augmentations, you should insert them into the **augmentation.py** file. And call them in the function *createModelInput()*

```
def createModelInput(audio,mel_transform, stft_trasform, augmentation=True):

    # Calcualate the first and the second augmentation
    if augmentation == True:
        audio = augmentation_1(audio)
        audio = augmentation_2(audio)
    
    # Create the augmented spectograms size [BATCH_SIZE, 3, 128, 126]
    spectograms = createSpectograms(audio, stft_trasform, mel_transform)
    spectograms = spectograms.to(device)

    return  spectograms, audio
```
# How to run the code
You can run these tests separately. Make sure you activate the env before. train.py is for training clar (semisupervised). supervised.py for suprevised training and selfsupervised.py for selfsupervised trianing.The hyperparamter can be found inside each file.

```
$python supervised.py  --epochs=101  --Batch_size=256 --wandb=enabled
$python selfsupervised.py  --epochs=101  --Batch_size=256 --wandb=enabled
$python train.py  --epochs=101  --Batch_size=256 --wandb=enabled --lab_percentage=1
```

# Spectograms, Melspectograms and Phase
The file Spectograms.py serve to achive the magnitude and phase spectogram, the melspectograms is achived using directly torchaudio. The function **createSpectograms()** will create the batch composed by stacked Magnitude, Mel and Phase spectograms.

# Logging
We higly reccomand to install wandb and remove mode="disabled" from the training pipeling if you desire to log the metrics, otherwise leave it disabled.



