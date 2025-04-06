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


def freuqency_perturbation(f, fs, alpha=2):
    n = np.linspace(1,fs,len(f))
    mean_amplitude = np.mean(f)
    noise_amplitude = mean_amplitude/np.power(n,alpha)
    noise = np.random.normal(0, 1, len(f))
    f = f + noise_amplitude * noise
    return abs(f)


class FreqencyAugumentation:
    def __init__(self, fs, mask_resolution, fft_resolution = 0.1):
        self.fs = fs
        self.max_resolution = mask_resolution
        self.fft_resolution = fft_resolution

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
        max_mask_length =int( self.max_resolution / self.fft_resolution )
        mask_start = np.random.randint(0,  high_cutoff - max_mask_length)
        f[..., mask_start:mask_start + max_mask_length] = 0
        return f
        
    def __call__(self, x):
        x = self._frequency_perturbation(x)
        x = self._frequency_masking(x)
        x = dropout(x, mask_rate=0.3)
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
        b, a = butter(5, [low, high], btype='bandstop')
        return b, a
    
    def _BandStopFilter(self, x, coeffs):
        b, a = coeffs
        return lfilter(b, a, x)
    
    def _dropout(self, x):
        mask = np.random.binomial(1, 1-self.mask_rate, x.shape[-1])
        return x * mask
    
    def _jitter(self, x):
        return x + np.random.normal(loc=0., scale=self.sigma, size=x.shape[-1])

    def __call__(self, x):
        #filter_coeff = np.random.choice(list(self.filters.keys()))
        #x = self._BandStopFilter(x, self.filters[filter_coeff])
        x = self._dropout(x)
        x = self._jitter(x)
        return x