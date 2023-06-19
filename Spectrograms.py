import torch
import torch.nn as nn
from torch.nn.functional import conv1d, conv2d
import numpy as np
from time import time
from scipy.signal import get_window
from scipy import signal
from scipy import fft
import warnings

# from librosa_filters import *  # Use it for PyPip, and PyTest

# from librosa_filters import * # Use it for debug

sz_float = 4  # size of a float
epsilon = 10e-8  # fudge factor for normalization


def broadcast_dim(x):
    """
    Auto broadcast input so that it can fits into a Conv1d
    """

    if x.dim() == 2:
        x = x[:, None, :]
    elif x.dim() == 1:
        # If nn.DataParallel is used, this broadcast doesn't work
        x = x[None, None, :]
    elif x.dim() == 3:
        pass
    else:
        raise ValueError("Only support input with shape = (batch, len) or shape = (len)")
    return x


## Kernal generation functions ##
def create_fourier_kernels(n_fft, win_length=None, freq_bins=None, fmin=50, fmax=6000, sr=44100,
                           freq_scale='linear', window='hann', verbose=True):
    """ This function creates the Fourier Kernel for STFT, Melspectrogram and CQT.
    Most of the parameters follow librosa conventions. Part of the code comes from
    pytorch_musicnet. https://github.com/jthickstun/pytorch_musicnet

    Parameters
    ----------
    n_fft : int
        The window size

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins

    fmin : int
        The starting frequency for the lowest frequency bin.
        If freq_scale is ``no``, this argument does nothing.

    fmax : int
        The ending frequency for the highest frequency bin.
        If freq_scale is ``no``, this argument does nothing.

    sr : int
        The sampling rate for the input audio. It is used to calculate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    freq_scale: 'linear', 'log', or 'no'
        Determine the spacing between each frequency bin.
        When 'linear' or 'log' is used, the bin spacing can be controlled by ``fmin`` and ``fmax``.
        If 'no' is used, the bin will start at 0Hz and end at Nyquist frequency with linear spacing.

    Returns
    -------
    wsin : numpy.array
        Imaginary Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``

    wcos : numpy.array
        Real Fourier Kernel with the shape ``(freq_bins, 1, n_fft)``

    bins2freq : list
        Mapping each frequency bin to frequency in Hz.

    binslist : list
        The normalized frequency ``k`` in digital domain.
        This ``k`` is in the Discrete Fourier Transform equation $$

    """

    if freq_bins == None: freq_bins = n_fft // 2 + 1
    if win_length == None: win_length = n_fft

    s = np.arange(0, n_fft, 1.)
    wsin = np.empty((freq_bins, 1, n_fft))
    wcos = np.empty((freq_bins, 1, n_fft))
    start_freq = fmin
    end_freq = fmax
    bins2freq = []
    binslist = []

    # num_cycles = start_freq*d/44000.
    # scaling_ind = np.log(end_freq/start_freq)/k

    # Choosing window shape

    window_mask = get_window(window, int(win_length), fftbins=True)
    #window_mask = pad_center(window_mask, n_fft)

    if freq_scale == 'linear':
        if verbose == True:
            print(f"sampling rate = {sr}. Please make sure the sampling rate is correct in order to"
                  f"get a valid freq range")
        start_bin = start_freq * n_fft / sr
        scaling_ind = (end_freq - start_freq) * (n_fft / sr) / freq_bins

        for k in range(freq_bins):  # Only half of the bins contain useful info
            # print("linear freq = {}".format((k*scaling_ind+start_bin)*sr/n_fft))
            bins2freq.append((k * scaling_ind + start_bin) * sr / n_fft)
            binslist.append((k * scaling_ind + start_bin))
            wsin[k, 0, :] = window_mask * np.sin(2 * np.pi * (k * scaling_ind + start_bin) * s / n_fft)
            wcos[k, 0, :] = window_mask * np.cos(2 * np.pi * (k * scaling_ind + start_bin) * s / n_fft)

    elif freq_scale == 'log':
        if verbose == True:
            print(f"sampling rate = {sr}. Please make sure the sampling rate is correct in order to"
                  f"get a valid freq range")
        start_bin = start_freq * n_fft / sr
        scaling_ind = np.log(end_freq / start_freq) / freq_bins

        for k in range(freq_bins):  # Only half of the bins contain useful info
            # print("log freq = {}".format(np.exp(k*scaling_ind)*start_bin*sr/n_fft))
            bins2freq.append(np.exp(k * scaling_ind) * start_bin * sr / n_fft)
            binslist.append((np.exp(k * scaling_ind) * start_bin))
            wsin[k, 0, :] = window_mask * np.sin(2 * np.pi * (np.exp(k * scaling_ind) * start_bin) * s / n_fft)
            wcos[k, 0, :] = window_mask * np.cos(2 * np.pi * (np.exp(k * scaling_ind) * start_bin) * s / n_fft)

    elif freq_scale == 'no':
        for k in range(freq_bins):  # Only half of the bins contain useful info
            bins2freq.append(k * sr / n_fft)
            binslist.append(k)
            wsin[k, 0, :] = window_mask * np.sin(2 * np.pi * k * s / n_fft)
            wcos[k, 0, :] = window_mask * np.cos(2 * np.pi * k * s / n_fft)
    else:
        print("Please select the correct frequency scale, 'linear' or 'log'")
    return wsin.astype(np.float32), wcos.astype(np.float32), bins2freq, binslist, window_mask




### --------------------------- Spectrogram Classes ---------------------------###
class STFT(torch.nn.Module):
    """This function is to calculate the short-time Fourier transform (STFT) of the input signal.
    Input signal should be in either of the following shapes.
        1. ``(len_audio)``
        2. ``(num_audio, len_audio)``
        3. ``(num_audio, 1, len_audio)``

    The correct shape will be inferred automatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.
    This class inherits from ``torch.nn.Module``, therefore, the usage is same as ``torch.nn.Module``.

    Parameters
    ----------
    n_fft : int
        The window size. Default value is 2048.

    freq_bins : int
        Number of frequency bins. Default is ``None``, which means ``n_fft//2+1`` bins

    hop_length : int
        The hop (or stride) size. Default value is 512.

    window : str
        The windowing function for STFT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'.

    freq_scale : 'linear', 'log', or 'no'
        Determine the spacing between each frequency bin. When `linear` or `log` is used,
        the bin spacing can be controlled by ``fmin`` and ``fmax``. If 'no' is used, the bin will
        start at 0Hz and end at Nyquist frequency with linear spacing.

    center : bool
        Putting the STFT keneral at the center of the time-step or not. If ``False``, the time
        index is the beginning of the STFT kernel, if ``True``, the time index is the center of
        the STFT kernel. Default value if ``True``.

    pad_mode : str
        The padding method. Default value is 'reflect'.

    fmin : int
        The starting frequency for the lowest frequency bin. If freq_scale is ``no``, this argument
        does nothing.

    fmax : int
        The ending frequency for the highest frequency bin. If freq_scale is ``no``, this argument
        does nothing.

    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.

    trainable : bool
        Determine if the STFT kenrels are trainable or not. If ``True``, the gradients for STFT
        kernels will also be caluclated and the STFT kernels will be updated during model training.
        Default value is ``False``

    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints

    device : str
        Choose which device to initialize this layer. Default value is 'cuda:0'

    Returns
    -------
    spectrogram : torch.tensor
        It returns a tensor of spectrograms.
            shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format``='Magnitude';
            shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format``='Complex' or 'Phase';

    Examples
    --------
    >>> spec_layer = Spectrogram.STFT()
    >>> specs = spec_layer(x)
    """

    def __init__(self, n_fft=2048, win_length=None, freq_bins=None, hop_length=None, window='hann',
                 freq_scale='no', center=True, pad_mode='reflect',
                 fmin=50, fmax=6000, sr=22050, trainable=False,
                 verbose=True, device='cuda:0'):

        super(STFT, self).__init__()

        # Trying to make the default setting same as librosa
        if win_length == None: win_length = n_fft
        if hop_length == None: hop_length = int(win_length // 4)

        self.trainable = trainable
        self.stride = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        self.trainable = trainable
        self.device = device
        self.pad_amount = self.n_fft // 2
        self.window = window
        self.win_length = win_length
        start = time()

        # Create filter windows for stft
        wsin, wcos, self.bins2freq, self.bin_list, window_mask = create_fourier_kernels(n_fft,
                                                                                        win_length=win_length,
                                                                                        freq_bins=freq_bins,
                                                                                        window=window,
                                                                                        freq_scale=freq_scale,
                                                                                        fmin=fmin,
                                                                                        fmax=fmax,
                                                                                        sr=sr,
                                                                                        verbose=verbose)
        # Create filter windows for inverse
        wsin_inv, wcos_inv, _, _, _ = create_fourier_kernels(n_fft,
                                                             win_length=win_length,
                                                             freq_bins=n_fft,
                                                             window='ones',
                                                             freq_scale=freq_scale,
                                                             fmin=fmin,
                                                             fmax=fmax,
                                                             sr=sr,
                                                             verbose=False)

        self.wsin = torch.tensor(wsin, dtype=torch.float, device=self.device)
        self.wcos = torch.tensor(wcos, dtype=torch.float, device=self.device)
        self.wsin_inv = torch.tensor(wsin_inv, dtype=torch.float, device=self.device)
        self.wcos_inv = torch.tensor(wcos_inv, dtype=torch.float, device=self.device)

        # Making all these variables nn.Parameter, so that the model can be used with nn.Parallel
        self.wsin = torch.nn.Parameter(self.wsin, requires_grad=self.trainable)
        self.wcos = torch.nn.Parameter(self.wcos, requires_grad=self.trainable)
        self.wsin_inv = torch.nn.Parameter(self.wsin_inv, requires_grad=self.trainable)
        self.wcos_inv = torch.nn.Parameter(self.wcos_inv, requires_grad=self.trainable)

        self.window_mask = torch.tensor(window_mask, device=self.device).unsqueeze(0).unsqueeze(-1)

        if verbose == True:
            print("STFT kernels created, time used = {:.4f} seconds".format(time() - start))
        else:
            pass

    def forward(self, x, output_format='Complex'):
        self.output_format = output_format
        self.num_samples = x.shape[-1]

        x = broadcast_dim(x)
        if self.center:
            if self.pad_mode == 'constant':
                padding = nn.ConstantPad1d(self.pad_amount, 0)

            elif self.pad_mode == 'reflect':
                if self.num_samples < self.pad_amount:
                    raise AssertionError("Signal length shorter than reflect padding length (n_fft // 2).")
                padding = nn.ReflectionPad1d(self.pad_amount)

            x = padding(x)
        spec_imag = conv1d(x, self.wsin, stride=self.stride)
        spec_real = conv1d(x, self.wcos, stride=self.stride)  # Doing STFT by using conv1d

        # remove redundant parts
        spec_real = spec_real[:, :self.freq_bins, :]
        spec_imag = spec_imag[:, :self.freq_bins, :]

        if output_format == 'Magnitude':
            self.output_format = 'Magnitude'
            spec = spec_real.pow(2) + spec_imag.pow(2)
            if self.trainable == True:
                return torch.sqrt(spec + 1e-8)  # prevent Nan gradient when sqrt(0) due to output=0
            else:
                return torch.sqrt(spec + 1e-8)

        elif output_format == 'Complex':
            self.output_format = 'Complex'
            return torch.stack((spec_real, -spec_imag), -1)  # Remember the minus sign for imaginary part

        elif output_format == 'Phase':
            self.output_format = 'Phase'
            return torch.atan2(-spec_imag + 0.0,
                               spec_real)  # +0.0 removes -0.0 elements, which leads to error in calculating phase

    def inverse(self, X, num_samples=-1):
        if len(X.shape) == 3 and self.output_format == 'Magnitude':
            return self.griffin_lim(X)
        elif len(X.shape) == 4 and self.output_format == "Complex":
            return self.__inverse(X, num_samples=num_samples)
        else:
            raise AssertionError("Only perform inverse function on Magnitude or Complex spectrogram.")

    def __inverse(self, X, is_window_normalized=True, num_samples=None):
        X_real, X_imag = X[:, :, :, 0], X[:, :, :, 1]

        # flip and extend beyond Nyquist frequency
        X_real_nyquist = torch.flip(X_real, [1])
        X_imag_nyquist = torch.flip(X_imag, [1])
        X_real_nyquist = X_real_nyquist[:, 1:-1, :]
        X_imag_nyquist = -X_imag_nyquist[:, 1:-1, :]

        X_real = torch.cat([X_real, X_real_nyquist], axis=1)
        X_imag = torch.cat([X_imag, X_imag_nyquist], axis=1)

        # broadcast dimensions to support 2D convolution
        X_real_bc = X_real.unsqueeze(1)
        X_imag_bc = X_imag.unsqueeze(1)
        wsin_bc = self.wsin_inv.unsqueeze(-1)
        wcos_bc = self.wcos_inv.unsqueeze(-1)

        a1 = conv2d(X_real_bc, wcos_bc, stride=(1, 1))
        b2 = conv2d(X_imag_bc, wsin_bc, stride=(1, 1))

        # compute real and imag part. signal lies in the real part
        real = a1 - b2
        real = real.squeeze(-2)

        # each time step contains reconstructed signal, hence we stitch them together and remove
        # the repeated segments due to overlapping windows during STFT when hop_size < n_fft
        if is_window_normalized:
            #             nonzero_indices = self.window_mask[0,:,0]>1e-6 # it doesn't work
            real /= self.window_mask

        real /= self.n_fft

        # It doesn't work if we keep the last time-step as a whole.
        # Since the erroneous part is on the LHS of the window, we better chop them out
        real_first = real[:, :, 0]  # get the complet first frame
        real_split = real[:, -self.stride:, 1:]  # remove redundant samples at overlapped parts

        # reshape output signal to 1D
        output_signal = torch.reshape(torch.transpose(real_split, 2, 1), (real_split.shape[0], -1))
        output_signal = torch.cat([real_first, output_signal], dim=-1)  # stich the first signal back

        output_signal = output_signal[:, self.pad_amount:]  # remove padding on LHS
        if num_samples:
            output_signal = output_signal[:, :num_samples]
        else:
            output_signal = output_signal[:, :-self.pad_amount]
        return output_signal

    def griffin_lim(self, X, maxiter=32, tol=1e-6, alpha=0.99, verbose=False, phase=None):
        # only use griffin lim when X is not in complex form
        if phase is None:
            phase = torch.rand_like(X) * 2 - 1
            phase = phase * np.pi
            phase[:, 0, :] = 0.0

        phase = nn.Parameter(phase)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam([phase], lr=9e-1)

        for idx in range(maxiter):
            optimizer.zero_grad()

            X_real, X_imag = X * torch.cos(phase), X * torch.sin(phase)
            X_cur = torch.stack([X_real, X_imag], dim=-1)
            inverse_signal = self.__inverse(X_cur, is_window_normalized=False)

            # Rebuild the spectrogram
            rebuilt_mag = self.forward(inverse_signal, output_format="Complex")
            rebuilt_mag = torch.sqrt(rebuilt_mag[:, :, :, 0].pow(2) + rebuilt_mag[:, :, :, 1].pow(2))
            loss = criterion(rebuilt_mag, X)
            loss.backward()
            optimizer.step()
            if verbose:
                print("Run: {}/{} MSE: {:.4}".format(idx + 1, maxiter, criterion(rebuilt_mag, X).item()))

        # Return the final phase estimates
        X_real, X_imag = X * torch.cos(phase), X * torch.sin(phase)
        X_cur = torch.stack([X_real, X_imag], dim=-1)
        return self.__inverse(X_cur, is_window_normalized=False)

