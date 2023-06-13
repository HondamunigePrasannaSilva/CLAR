"""
    Audio augmentation file

    1. Frequency Transformations:

    (a) Pitch Shift
    (b) Noise Injection
    
    2. Temporal Transformations
    
    (a) Fade in/out
    (b) Time Masking
    (c) Time Shift (TS): randomly shifts the audio samples forwards or backwards. Samples that 
        roll beyond the last position are re-introduced at the first position (rollover). The 
        degree and direction of the shifts were randomly selected for each audio. The maximum 
        degree that could be shifted was half of the audio signal, while, the minimum was when 
        no shift applied to the signal.
    (d) Time Stretching (TST): slows down or speeds up the audio sample (while keeping the pitch unchanged). 
        In this approach we transformed the signal by first computing the STFT of the signal, stretching 
        it using a phase vocoder, and computing the inverse STFT to reconstruct the time domain signal.
        Following those transformations, we down-sampled or cropped the signal to match the same number 
        of samples as the input signal. When the rate of stretching was greater than 1, the signal was 
        sped up. Otherwise when the rate of stretching was less than 1, then the signal was slowed down. 
        The rate of time stretching was randomized for each audio with range values of [0.5, 1.5].

"""
from typing import Any
import torchaudio
import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
import random
import torch
from torchmetrics import SignalNoiseRatio

def pitchshift(audio, SAMPLE_RATE=16000, shift = 5):

    """
    Pitch Shift (PS): randomly raises or lowers the pitch of the audio signal.\n
    Based on experimental observation,we found the range of pitch shifts that main-tained\n
    the overall coherency of the input audio was in the range [-15, 15] semitones. 

    Attributes:
    - :param audio: audio tensor
    - :param SAMPLE_RATE: Sample rate, default=16000
    - :param shift: Pitch shift 
    - :return: describe what it returns
    """
    assert audio != None, "audio should not be None"
    transform = transforms.PitchShift(sample_rate=SAMPLE_RATE, n_steps=shift)
    waveform_shift = transform(audio)
    return waveform_shift


def fade_in_out(audio):
    """
    Fade in/out (FD): gradually increases/decreases the intensity of the audio in the\n
    beginning/end of the audio signal.\n
    The degree of the fade was either linear, logarithmic or exponential (applied\n
    with uniform probability of 1/3). The size of the fade for either side of the\n
    audio signal could at maximum reach half of the audio signal. The size of the\n
    fade was another random parameter picked for each sample.
    """
    assert audio != None, "audio should not be None"
    _fade_shape = ['linear', 'logarithmic', 'exponential']
    _fade_size = [i for i in range(1, int(audio.shape[1]/2))]

    transform = transforms.Fade(fade_in_len=random.choice(_fade_size), fade_out_len=random.choice(_fade_size), fade_shape=random.choice(_fade_shape))
    waveform_fade_in_out = transform(audio)
    return waveform_fade_in_out


def _noise_injection(audio):
    """
    Noise Injection: mix the audio signal with random white, brown and pink noise.\n
    In our implementation, the intensity of the noise signal was randomly selected based\n
    on the strength of signal-to-noise ratio. Applied white, brown, or pink depending\n
    on an additional random parameter sampled from uniform distribution (Mixed Noise).
    """
    assert audio != None, "audio should not be None"

    white_noise, _ = torchaudio.load("SpeechCommands/speech_commands_v0.02/_background_noise_/white_noise.wav")
    pink_noise, _ = torchaudio.load("SpeechCommands/speech_commands_v0.02/_background_noise_/pink_noise.wav")

    noise = [white_noise[0][:16000][None,:], pink_noise[0][:16000][None,:] ]

    noise = random.choice(noise)  #select random noise 
    transform = transforms.AddNoise()
    snr = SignalNoiseRatio()
    
    waveform_with_noise = transform(audio, noise)
    return waveform_with_noise



def _time_masking(audio):
    """
    Time masking:given an audio signal, in this transformation we randomly select a small\n
    segment of the full signal and set the signal values in that segment to normal noise or a\n 
    constant value. In our implementation, we not only randomly selected the location of the\n
    masked segment but also we randomly selected the size of the segment. The size of the \n
    masked segment was set to maximally be 1/8 of the input signal.
    """
    assert audio != None, "audio should not be None"

    masked_loc = [i for i in range(audio.shape[1])]
    masked_len = [i for i in range(1, int(audio.shape[1]/8))]

    return 



# add the augmentation function here!
augmentation = {
     'pitchshift':lambda audio:pitchshift(audio),
      'fade_in_out':lambda audio:fade_in_out(audio),
     'noise_injection':lambda audio:_noise_injection(audio),
     'time_masking':lambda audio:_time_masking(audio)
}


class Augment:

    """
    Module that apply two augmentation:
    """

    def __init__(self, augmentation_1, augmentation_2):

        assert augmentation_1 in augmentation, f"Augmentation: {augmentation_1} not found in the list"
        assert augmentation_2 in augmentation, f"Augmentation: {augmentation_2} not found in the list"

        self.aug_1 = augmentation[augmentation_1]
        self.aug_2 = augmentation[augmentation_2]


    def __call__(self,audio):

        audio_1 = self.aug_1(audio)
        audio_2 = self.aug_2(audio_1)

        return  audio_2


import numpy as np

def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise*noise_factor
    return augmented_signal



