import numpy as np
from scipy.signal import butter, lfilter

def dropout(x, mask_rate=0.1):
    mask = np.random.binomial(1, 1-mask_rate, x.shape)
    return x * mask

def jitter(x, sigma=0.8):
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def gaussian_noise(x, sigma=0.1):
    noise = np.random.normal(loc=0., scale=sigma, size=x.shape)
    return x + noise

def mask(x, mask_ratio=0.1):
    start = np.random.randint(0, x.shape[-1] - int(mask_ratio * x.shape[-1]))
    end = start + int(mask_ratio * x.shape[-1])
    x[..., start:end] = 0
    return x

class FreqencyAugumentation:
    def __init__(self, fs, masked_freq, fft_resolution = 0.1, dropout_rate=0.3):
        self.fs = fs
        self.masked_freq = masked_freq
        self.fft_resolution = fft_resolution
        self.dropout_rate = dropout_rate

    def _frequency_perturbation(self, f):
        mean_amplitude = np.mean(f, axis=-1)
        n = np.linspace(1,self.fs,f.shape[-1])
        noise_amplitude = mean_amplitude
        noise = np.random.normal(0, 1, f.shape[-1])
        scaled_noise = noise_amplitude * noise
        f = f + scaled_noise[None,:]
        return f
    
    def _frequency_masking(self, f):
        high_cutoff = int(0.5 *  f.shape[-1])
        max_mask_length =int( self.masked_freq / self.fft_resolution )
        mask_start = np.random.randint(0,  high_cutoff - max_mask_length)
        f[..., mask_start:mask_start + max_mask_length] = 0
        return f
    
    def _dropout(self, f):
        mask = np.random.binomial(1, 1-self.dropout_rate, f.shape[-1])
        return f * mask
        
    def __call__(self, x, perturbation=True, masking=True):
        if perturbation:
            x = self._frequency_perturbation(x)
        if masking:
            x = self._frequency_masking(x)
            x = dropout(x, mask_rate=self.dropout_rate)
        return x
    
class TimeAugumentation:
    def __init__(self, fs, mask_rate, sigma):
        '''
        fs: int sampling rate
        mask_rate: float, mask rate
        sigma: float, jitter sigma
        '''
        self.mask_rate = mask_rate
        self.sigma = sigma
        self.initialize(fs)

    def initialize(self, fs):
        self.filters = {}
        for i in range(1, 30, 5):
            self.filters[i] = self.compute_filter_coefficients(fs, [i, i+5])

    def compute_filter_coefficients(self, fs, cutoffs):
        nyquist = 0.5 * fs
        low = cutoffs[0] / nyquist
        high = cutoffs[1] / nyquist
        b, a = butter(2, [low, high], btype='bandstop', analog=False)
        return b, a
    
    def _BandStopFilter(self, x, coeffs):
        b, a = coeffs
        return lfilter(b, a, x,axis=-1)
    
    def _dropout(self, x):
        mask = np.random.binomial(1, 1-self.mask_rate, x.shape[-1])
        return x * mask
    
    def _jitter(self, x):
        return x + np.random.normal(loc=0., scale=self.sigma, size=x.shape[-1])

    def __call__(self, x, bandstop=True, dropout=True, jitter=True):
        if bandstop:
            filter_coeff = np.random.choice(list(self.filters.keys()))
            x = self._BandStopFilter(x, self.filters[filter_coeff])
        if dropout:
            x = self._dropout(x)
            x = mask(x, mask_ratio=self.mask_rate)
        if jitter:
            x = self._jitter(x)
        return x