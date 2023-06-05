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

    _fade_shape = ['linear', 'logarithmic', 'exponential']
    _fade_size = [i for i in range(1, int(audio.shape[1]/2))]

    transform = transforms.Fade(fade_in_len=random.choice(_fade_size), fade_out_len=random.choice(_fade_size), fade_shape=random.choice(_fade_shape))
    waveform_fade_in_out = transform(audio)
    return waveform_fade_in_out


def noise_injection(audio):
    """
    Noise Injection: mix the audio signal with random white, brown and pink noise.\n
    In our implementation, the intensity of the noise signal was randomly selected based\n
    on the strength of signal-to-noise ratio. Applied white, brown, or pink depending\n
    on an additional random parameter sampled from uniform distribution (Mixed Noise).
    """
    white_noise, _ = torchaudio.load("SpeechCommands/speech_commands_v0.02/_background_noise_/white_noise.wav")
    pink_noise, _ = torchaudio.load("SpeechCommands/speech_commands_v0.02/_background_noise_/pink_noise.wav")
    noise = [white_noise[0][:16000][None,:], pink_noise[0][:16000][None,:] ]

    noise = random.choice(noise)  #select random noise 
    transform = transforms.AddNoise()
    snr = SignalNoiseRatio()
    print(snr(audio,noise))
    
    waveform_with_noise = transform(audio, noise)
    return waveform_with_noise



def time_masking(audio):
    """
    Time masking:given an audio signal, in this transformation we randomly select a small\n
    segment of the full signal and set the signal values in that segment to normal noise or a\n 
    constant value. In our implementation, we not only randomly selected the location of the\n
    masked segment but also we randomly selected the size of the segment. The size of the \n
    masked segment was set to maximally be 1/8 of the input signal.
    """

    masked_loc = [i for i in range(audio.shape[1])]
    masked_len = [i for i in range(1, int(audio.shape[1]/8))]

def getTransform(title):
    if(title == "pitchshift"):
        trasform = pitchshift
    if(title == "fade_in_out"):
        trasform = fade_in_out
    if(title == "noise_injection"):
        trasform = noise_injection
    if(title == "time_masking"):
        trasform = time_masking
    

def transform(augmentation, input):
    """
    augmentation: augmentation name
    input with batch -> [32, 1, 16000]
    """
    augmentedAudio = torch.tensor(size = [input.shape[0], input.shape[1], input.shape[2]])

    transform = getTransform(augmentation)

    for i in range(input.shape[0]):
        augmentedAudio[i] = transform(input[i])
    
    return augmentedAudio





