import numpy as np
import scipy.signal
from scipy.signal import find_peaks


def separate_breath(bcg: np.ndarray, SR: int) -> np.ndarray:
    filter_coeffs = scipy.signal.butter(5, 0.7, 'low', fs=SR)
    
    return scipy.signal.filtfilt(*filter_coeffs, bcg)

def separate_heart(bcg: np.ndarray, SR: float) -> np.ndarray:
    filter_coeffs = scipy.signal.butter(5, [5, 15], 'band', fs=SR)

    return scipy.signal.filtfilt(*filter_coeffs, bcg)

def normalize(signal: np.ndarray, window_size: int = 70) -> np.ndarray:
    normalized_signal = np.zeros_like(signal)
    
    for i in range(len(signal)):
        start_index = max(0, i - window_size + 1)
        window = signal[start_index:i+1]
        min_val = np.min(window)
        max_val = np.max(window)
        if max_val != min_val:
            normalized_signal[i] = (signal[i] - min_val) / (max_val - min_val)
        else:
            normalized_signal[i] = 0.5

    return normalized_signal

def calculate_peaks(signal: np.ndarray, window_size: int = 10, threshold: float = 0.75) -> np.ndarray:
    max_min_diff = np.zeros_like(signal)
    upto = np.zeros_like(signal)
    result = np.zeros_like(signal)

    peaks = []
    intervals = []
    
    for i in range(len(signal)):
        if i >= window_size:
            window = signal[i-window_size+1:i+1]
            max_val = np.max(window)
            min_val = np.min(window)
            max_min_diff[i] = max_val - min_val
            if max_min_diff[i] >= threshold:
                if upto[i-1] == 0:
                    upto[i] = max_val
                    peaks.append(i)
    
    if len(peaks) > 1:
        intervals = np.diff(peaks)
        avg_interval = np.mean(intervals)

        for i in range(len(peaks)):
            if i == 0 or intervals[i-1] >= avg_interval / 2:
                result[peaks[i]] = upto[peaks[i]]

    return result

def calculate_heartrate(result: np.ndarray, SR: int):
    peaks = np.where(result > 0)[0]
    intervals = np.diff(peaks) / SR
    bpm = 60.0 / np.mean(intervals)
    
    return bpm

def find_minima_and_calculate_rr(signal: np.ndarray, SR: int):
    minima, _ = find_peaks(-signal)
    num_inflection_points = len(minima)
    time_duration = len(signal) / SR
    breaths = num_inflection_points
    respiratory_rate = breaths / time_duration * 60
    
    return respiratory_rate, minima
